import os
import pickle
import pandas as pd
from gurobipy import Model, GRB, quicksum

print("=" * 80)
print("DEPOT ROUTING - OPTION A: PURE 4D z[i,j,t,day]")
print("Single variable captures: route i from depot j by mode t on day")
print("=" * 80)

# ============================================================
# 1) LOAD DATA
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[step 1] loading data")

with open("model_inputBGUXY.pkl", "rb") as f:
    model_data = pickle.load(f)

I_all = list(model_data["sets"]["I"])
J = list(model_data["sets"]["J"])
T = list(model_data["sets"]["T"])
D = model_data["parameters"]["D"]
d = model_data["parameters"]["d"]
S = model_data["parameters"]["service_time"]

print(f"[ok] |I|={len(I_all)}, |J|={len(J)}, |T|={len(T)}")

# ============================================================
# 1B) SPLIT BY DAY
# ============================================================
T_days = ["TueThur", "WedFri"]
xy_codes = model_data["parameters"].get("xy", {})

I_by_day = {"TueThur": [], "WedFri": []}
for i in I_all:
    xy = xy_codes.get(i)
    if xy == "X":
        I_by_day["TueThur"].append(i)
    elif xy == "Y":
        I_by_day["WedFri"].append(i)

print(f"[ok] Tue/Thu: {len(I_by_day['TueThur'])}, Wed/Fri: {len(I_by_day['WedFri'])}")

# ============================================================
# 2) PARAMETERS
# ============================================================
c_s = 0.42
f_j = {j: 100 for j in J}
C_d = 10000

c_t_s = {"FootBike": 0.52, "Car": 95.53, "EBikeCart": 14.58, "Moped": 18.54}
c_t = {"FootBike": 0.0, "Car": 1.07, "EBikeCart": 0.03, "Moped": 0.22}
v_t = {"FootBike": 13, "Car": 35, "EBikeCart": 17.5, "Moped": 22.5}
R_t = {"FootBike": 3.0, "Car": 12, "EBikeCart": 3.5, "Moped": 4.5}
w = 23.30

# ============================================================
# 3) CANDIDATE DEPOTS
# ============================================================
print("[step 3] selecting candidate depots")
N_DEPOTS = 20
J_cand = {}
for i in I_all:
    cand = [(j, d[(i, j)]) for j in J if (i, j) in d]
    cand.sort(key=lambda x: x[1])
    J_cand[i] = [j for (j, _) in cand[:N_DEPOTS]]

bad_routes = [i for i in I_all if len(J_cand[i]) == 0]
if bad_routes:
    I_all = [i for i in I_all if i not in bad_routes]
    for day in T_days:
        I_by_day[day] = [i for i in I_by_day[day] if i not in bad_routes]
    D = {i: D[i] for i in I_all}
    J_cand = {i: J_cand[i] for i in I_all}

print(f"[ok] routes: {len(I_all)}")

# ============================================================
# 4) BUILD IJTS (valid route-depot-mode combos for each day)
# ============================================================
print("[step 4] building valid (i,j,t) combinations")
IJTS = {}
for day in T_days:
    I = I_by_day[day]
    ijts = []
    for i in I:
        for j in J_cand[i]:
            # Check BOTH distance directions exist (both must be in preprocessed d)
            if (j, i) in d and (i, j) in d:
                for t in T:
                    # ONLY add if service time also exists (skip incomplete combos)
                    if (i, j, t) in S:
                        ijts.append((i, j, t))
    IJTS[day] = ijts
    print(f"[ok] {day}: {len(ijts)} (i,j,t) combos")

# ============================================================
# 5) MODEL & VARIABLES
# ============================================================
print("[step 5] building model")
m = Model("VRP_TwoDayClean_OptionA")

# Gurobi tuning
m.Params.Threads = 0
m.Params.MIPFocus = 1
m.Params.Heuristics = 0.2
m.Params.Cuts = 2
m.Params.Presolve = 2
m.Params.NodeMethod = 1
m.Params.OutputFlag = 1

# Global variables
y = m.addVars(J, vtype=GRB.BINARY, name="y")  # depot open
n = m.addVars(T, vtype=GRB.INTEGER, lb=0, name="n")  # total fleet size

# ====== SINGLE 4D VARIABLE ======
# z[i,j,t,day] = 1 if route i served from depot j by mode t on day
z = {}
for day in T_days:
    ijts = IJTS[day]
    z_day = m.addVars(ijts, vtype=GRB.BINARY, name=f"z_{day}")
    for (i, j, t) in ijts:
        z[i, j, t, day] = z_day[i, j, t]

# Vehicle allocation (derived from z)
n_jt = {}
for day in T_days:
    n_jt_day = m.addVars(J, T, vtype=GRB.INTEGER, lb=0, name=f"n_jt_{day}")
    for j in J:
        for t in T:
            n_jt[j, t, day] = n_jt_day[j, t]

# Time tracking
g = {}
for day in T_days:
    g_day = m.addVars(T, lb=0, name=f"g_{day}")
    for t in T:
        g[t, day] = g_day[t]

m.update()
print(f"[ok] variables: {m.NumVars:,}")
print(f"    z: ~{len(IJTS['TueThur']) + len(IJTS['WedFri']):,} (route-depot-mode)")
print(f"    y: {len(J)} (depot open)")
print(f"    n: {len(T)} (fleet size)")

# ============================================================
# 6) OBJECTIVE
# ============================================================
# Fixed costs
depot_fixed = 2 * quicksum(f_j[j] * y[j] for j in J)
vehicle_fixed = 2 * quicksum(c_t_s[t] * n[t] for t in T)

# Variable costs
storage_cost = quicksum(
    c_s * D[i] * z[i, j, t, day]
    for day in T_days
    for (i, j, t) in IJTS[day]
)

labor_cost = quicksum((c_t[t] + w) * g[t, day] for t in T for day in T_days)

m.setObjective(depot_fixed + vehicle_fixed + storage_cost + labor_cost, GRB.MINIMIZE)
print("[step 6] objective set")

# ============================================================
# 7) CONSTRAINTS
# ============================================================
print("[step 7] adding constraints")
count = 0

# Each route served exactly once per day
for day in T_days:
    I = I_by_day[day]
    ijts = IJTS[day]

    m.addConstrs(
        (quicksum(z[i2, j2, t2, day] for (i2, j2, t2) in ijts if i2 == i) == 1
         for i in I),
        name=f"{day}_route_once"
    )
    count += len(I)

print(f"  [ok] route visit: {count}")

# Depot must be open
for day in T_days:
    ijts = IJTS[day]
    m.addConstrs(
        (z[i, j, t, day] <= y[j] for (i, j, t) in ijts),
        name=f"{day}_depot_open"
    )
    count += len(ijts)

print(f"  [ok] depot open: {count}")

# Depot capacity
for day in T_days:
    ijts = IJTS[day]

    for j in J:
        m.addConstr(
            quicksum(
                D[i2] * z[i2, j2, t2, day]
                for (i2, j2, t2) in ijts
                if j2 == j
            ) <= C_d * y[j],
            name=f"{day}_cap_{j}"
        )
    count += len(J)

print(f"  [ok] depot capacity: {count}")

# Vehicle allocation per depot per day
for day in T_days:
    ijts = IJTS[day]

    for j in J:
        for t in T:
            m.addConstr(
                n_jt[j, t, day] == quicksum(
                    z[i2, j2, t2, day]
                    for (i2, j2, t2) in ijts
                    if j2 == j and t2 == t
                ),
                name=f"{day}_n_jt_{j}_{t}"
            )
    count += len(J) * len(T)

print(f"  [ok] vehicle allocation: {count}")

# Fleet sizing (max over both days)
for t in T:
    m.addConstr(n[t] >= quicksum(n_jt[j, t, "TueThur"] for j in J), name=f"fleet_tue_{t}")
    m.addConstr(n[t] >= quicksum(n_jt[j, t, "WedFri"] for j in J), name=f"fleet_wed_{t}")
count += 2 * len(T)

print(f"  [ok] fleet sizing: {count}")

# Time accounting and uptime limits
for day in T_days:
    ijts = IJTS[day]

    for t in T:
        # Compute total time for mode t on day
        travel = quicksum(
            z[i2, j2, t2, day] * (d[(j2, i2)] + d[(i2, j2)]) / v_t[t2]
            for (i2, j2, t2) in ijts if t2 == t
        )
        service = quicksum(
            z[i2, j2, t2, day] * S[(i2, j2, t2)]
            for (i2, j2, t2) in ijts if t2 == t
        )
        m.addConstr(g[t, day] >= travel + service, name=f"{day}_time_{t}")

        # # Uptime limit
        # m.addConstr(g[t, day] <= n[t] * R_t[t], name=f"{day}_uptime_{t}")
        # 30 <= 10 * 4.5
        # z [route a, depot x, fiets, tue] = 5 hou
        # z [route b, depot x, fiets, tue]
        # z [route b, depot x, fiets, tue]
        # z [route b, depot x, fiets, tue]
        # z [route b, depot x, fiets, tue]

    for (i, j, t) in ijts:
        t_travel = (d[(j, i)] + d[(i, j)]) / v_t[t]  # Not in S!
        t_service = S[(i, j, t)]  # From S
        t_total = t_travel + t_service
        m.addConstr(
            z[i, j, t, day] * t_total <= R_t[t],
            name=f"{day}_route_uptime_{i}_{j}_{t}"
        )
    count += 2 * len(T)

print(f"  [ok] time constraints: {count}")

print(f"[ok] total constraints: {m.NumConstrs:,}")

# ============================================================
# 8) SOLVE
# ============================================================
print("[step 8] optimizing")
m.Params.MIPGap = 0.01
m.Params.TimeLimit = 3600
m.optimize()

print(f"[status] {m.Status}")

if m.Status == GRB.INFEASIBLE:
    m.computeIIS()
    m.write("model_iis.ilp")
    exit()

# ============================================================
# 9) RESULTS & EXCEL EXPORT
# ============================================================
if m.SolCount > 0:
    print(f"\n{'='*60}")
    print(f"Objective value: €{m.ObjVal:,.2f}")
    print(f"Optimality gap: {m.MIPGap:.2%}")
    print(f"{'='*60}")

    # ========== 1) MODEL SUMMARY ==========
    model_summary = [{
        "objective_value_2days": round(m.ObjVal, 2),
        "objective_value_daily": round(m.ObjVal / 2, 2),
        "objective_value_annual": round(m.ObjVal * (365 / 2), 2),
        "optimality_gap": round(m.MIPGap, 4),
        "status": m.Status,
        "num_routes_total": len(I_all),
        "num_routes_tueThur": len(I_by_day["TueThur"]),
        "num_routes_wedFri": len(I_by_day["WedFri"]),
        "num_depots": len(J),
        "num_modes": len(T),
        "num_variables": m.NumVars,
        "num_constraints": m.NumConstrs,
    }]

    # ========== 2) COST BREAKDOWN ==========
    vehicle_fixed_cost_2days = 2 * sum(c_t_s[t] * n[t].X for t in T)
    depot_cost_2days = 2 * sum(f_j[j] * y[j].X for j in J)
    labor_electricity_cost = sum((c_t[t] + w) * g[t, day].X for t in T for day in T_days)
    storage_handling_cost = sum(
        c_s * D[i] * z[i, j, t, day].X
        for day in T_days
        for (i, j, t) in IJTS[day]
    )

    cost_rows = [{
        "cost_depots_2days": round(depot_cost_2days, 2),
        "cost_vehicles_fixed_2days": round(vehicle_fixed_cost_2days, 2),
        "cost_storage_handling": round(storage_handling_cost, 2),
        "cost_labor_electricity": round(labor_electricity_cost, 2),
        "total_cost_2days": round(m.ObjVal, 2),
        "cost_daily_average": round(m.ObjVal / 2, 2),
        "cost_annual": round(m.ObjVal * (365 / 2), 2),
    }]

    # ========== 3) DAY SUMMARY ==========
    day_summary = []
    for day in T_days:
        I = I_by_day[day]
        total_routes = len(I)
        total_demand = sum(D[i] for i in I)
        depots_open = len([j for j in J if y[j].X > 0.5])
        total_work = sum(g[t, day].X for t in T)
        day_summary.append({
            "day": day,
            "num_routes": total_routes,
            "total_demand": total_demand,
            "depots_open": depots_open,
            "total_work_hours": round(total_work, 2),
        })
        print(f"\n[{day}]")
        print(f"  Routes: {total_routes}")
        print(f"  Demand: {total_demand} bags")
        print(f"  Work hours: {total_work:.2f}")
        print(f"  Depots: {depots_open}")

    # ========== 4) MODE USAGE (per day) ==========
    mode_rows = []
    for t in T:
        for day in T_days:
            work_time = g[t, day].X
            routes_count = sum(1 for (i, j, t2) in IJTS[day] if t2 == t and z[i, j, t, day].X > 0.5)
            vehicles_deployed = sum(int(round(n_jt[j, t, day].X)) for j in J)
            max_capacity = vehicles_deployed * R_t[t] if vehicles_deployed > 0 else 0.0
            mode_rows.append({
                "day": day,
                "mode": t,
                "vehicles_total": int(round(n[t].X)),
                "vehicles_deployed": vehicles_deployed,
                "num_routes": routes_count,
                "work_time_hours": round(work_time, 2),
                "max_capacity_hours": round(max_capacity, 2),
                "utilization_pct": round(100 * work_time / max_capacity, 1) if max_capacity > 0 else 0.0,
            })

    # ========== 5) DEPOT USAGE (per day) ==========
    depot_rows = []
    for j in J:
        for day in T_days:
            assigned_routes = [i for (i, j2, t) in IJTS[day] if j2 == j and z[i, j, t, day].X > 0.5]
            demand = sum(D[i] for i in assigned_routes)
            depot_rows.append({
                "day": day,
                "depot": j,
                "open": int(round(y[j].X)),
                "num_routes": len(assigned_routes),
                "total_demand": demand,
            })

    # ========== 6) DEPOT-MODE MATRIX ==========
    depot_mode_rows = []
    for day in T_days:
        for j in J:
            for t in T:
                vehicles_at_depot = int(round(n_jt[j, t, day].X))
                depot_mode_rows.append({
                    "day": day,
                    "depot": j,
                    "mode": t,
                    "vehicles_allocated": vehicles_at_depot,
                    "depot_open": int(round(y[j].X)),
                })

    # ========== 7) ROUTE-LEVEL DETAILS ==========
    route_rows = []
    for day in T_days:
        ijts = IJTS[day]
        for (i, j, t) in ijts:
            if z[i, j, t, day].X > 0.5:
                t_travel = (d[(j, i)] + d[(i, j)]) / v_t[t]
                t_service = S[(i, j, t)]
                t_total = t_travel + t_service
                cost_storage = c_s * D[i]
                cost_labor = (w + c_t[t]) * t_total
                cost_route_total = cost_storage + cost_labor

                route_rows.append({
                    "day": day,
                    "route": i,
                    "depot": j,
                    "mode": t,
                    "demand": D[i],
                    "service_time_hours": round(t_service, 2),
                    "travel_hours_total": round(t_travel, 2),
                    "total_time_hours": round(t_total, 2),
                    "cost_storage_handling": round(cost_storage, 2),
                    "cost_labor": round(cost_labor, 2),
                    "cost_route_total": round(cost_route_total, 2),
                })

    # ========== WRITE TO EXCEL ==========
    out_path = os.path.join(SCRIPT_DIR, "solution_OptionA.xlsx")
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        pd.DataFrame(model_summary).to_excel(writer, sheet_name="ModelSummary", index=False)
        pd.DataFrame(cost_rows).to_excel(writer, sheet_name="CostBreakdown", index=False)
        pd.DataFrame(day_summary).to_excel(writer, sheet_name="DaySummary", index=False)
        pd.DataFrame(mode_rows).to_excel(writer, sheet_name="ModeUsage", index=False)
        pd.DataFrame(depot_rows).to_excel(writer, sheet_name="DepotUsage", index=False)
        pd.DataFrame(depot_mode_rows).to_excel(writer, sheet_name="DepotModeMatrix", index=False)
        pd.DataFrame(route_rows).to_excel(writer, sheet_name="Routes", index=False)

    print(f"\n[ok] solution written to: {out_path}")

    print("\n" + "=" * 80)
    print("OBJECTIVE VALUE BREAKDOWN:")
    print("=" * 80)
    print(f"Vehicles (2 days, fixed): €{vehicle_fixed_cost_2days:,.2f}")
    print(f"Depots (2 days, fixed):   €{depot_cost_2days:,.2f}")
    print(f"Storage/Handling (c_s):   €{storage_handling_cost:,.2f}")
    print(f"Labor + Electricity:      €{labor_electricity_cost:,.2f}")
    print(f"  " + "-" * 20)
    print(f"TOTAL (2 days):           €{m.ObjVal:,.2f}")
    print(f"Daily average:            €{m.ObjVal / 2:,.2f}")
    print(f"Annual (scaled):          €{m.ObjVal * (365 / 2):,.2f}")
    print("=" * 80)

else:
    print("[warn] no feasible solution — nothing exported")

print("[✓] export complete!")
