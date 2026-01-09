import os
import pickle
import pandas as pd
from gurobipy import Model, GRB, quicksum

# =============================================================================
# depot routing milp with two-day structure and vehicle reuse
#
# this script assigns each delivery route to:
# - exactly one depot
# - exactly one transport mode
# - on its predefined service day (X or Y)
#
# vehicle reuse is modeled through time capacity at the depot level instead of
# explicit vehicle-route assignment. one vehicle can serve multiple routes
# as long as total travel + service time fits within the working day.
#
# comments are written deliberately plain and compact, similar to how you would
# normally comment your own working code.
# =============================================================================

print("=" * 80)
print("DEPOT ROUTING - OPTION A")
print("two-day structure with vehicle reuse via depot time capacity")
print("=" * 80)

# =============================================================================
# 1) load input data
# =============================================================================
print("[step 1] loading model input")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_NAME = "model_inputBGUXY.pkl"

with open(os.path.join(SCRIPT_DIR, PKL_NAME), "rb") as f:
    model_data = pickle.load(f)

# sets
I_all = list(model_data["sets"]["I"])   # delivery routes / clusters
J = list(model_data["sets"]["J"])       # candidate depots
T = list(model_data["sets"]["T"])       # transport modes

# parameters
D = model_data["parameters"]["D"]              # demand per route
d = model_data["parameters"]["d"]              # distances (route-depot)
S = model_data["parameters"]["service_time"]   # service times per (i,j,t)

print(f"[ok] routes: {len(I_all)}")
print(f"[ok] depots: {len(J)}")
print(f"[ok] modes: {len(T)}")

# =============================================================================
# 1b) split routes by service day
# =============================================================================
print("[step 1b] splitting routes by service day")

T_days = ["X", "Y"]
xy_codes = model_data["parameters"].get("xy", {})

I_by_day = {"X": [], "Y": []}
for i in I_all:
    if xy_codes.get(i) == "X":
        I_by_day["X"].append(i)
    elif xy_codes.get(i) == "Y":
        I_by_day["Y"].append(i)

print(f"[ok] X routes: {len(I_by_day['X'])}")
print(f"[ok] Y routes: {len(I_by_day['Y'])}")

# =============================================================================
# 2) cost and operational parameters
# =============================================================================
print("[step 2] setting cost and operational parameters")

c_s = 0.42                                    # storage/handling cost per bag
f_j = {j: 50 for j in J}                     # fixed depot cost
C_d = 10000                                   # max demand per depot per day

c_t_s = {                                    # fixed cost per vehicle
    "FootBike": 0.52,
    "Car": 95.53,
    "EBikeCart": 14.58,
    "Moped": 18.54,
}

c_t = {                                      # variable cost per hour
    "FootBike": 0.0,
    "Car": 1.07,
    "EBikeCart": 0.03,
    "Moped": 0.22,
}

v_t = {                                      # average speed
    "FootBike": 13,
    "Car": 35,
    "EBikeCart": 17.5,
    "Moped": 22.5,
}

R_t = {                                      # max working hours per vehicle
    "FootBike": 3.0,
    "Car": 12,
    "EBikeCart": 3.5,
    "Moped": 4.5,
}

w = 23.30                                    # wage per hour

# =============================================================================
# 3) select candidate depots per route
# =============================================================================
print("[step 3] selecting nearest depots per route")

N_DEPOTS = 20
J_cand = {}

for i in I_all:
    cand = [(j, d[(i, j)]) for j in J if (i, j) in d]
    cand.sort(key=lambda x: x[1])
    J_cand[i] = [j for (j, _) in cand[:N_DEPOTS]]

# remove routes without any feasible depot
bad_routes = [i for i in I_all if len(J_cand[i]) == 0]
if bad_routes:
    print(f"[warn] removing {len(bad_routes)} routes without depot links")
    I_all = [i for i in I_all if i not in bad_routes]
    for day in T_days:
        I_by_day[day] = [i for i in I_by_day[day] if i not in bad_routes]
    D = {i: D[i] for i in I_all}
    J_cand = {i: J_cand[i] for i in I_all}

print(f"[ok] routes remaining after filtering: {len(I_all)}")

# =============================================================================
# 4) build valid (i,j,t) combinations per day
# =============================================================================
print("[step 4] building valid route–depot–mode combinations")

IJTS = {}
for day in T_days:
    ijts = []
    for i in I_by_day[day]:
        for j in J_cand[i]:
            if (i, j) in d and (j, i) in d:
                for t in T:
                    if (i, j, t) in S:
                        ijts.append((i, j, t))
    IJTS[day] = ijts
    print(f"[ok] {day}: {len(ijts)} valid combinations")

# =============================================================================
# 5) build optimization model and variables
# =============================================================================
print("[step 5] building optimization model")

m = Model("VRP_TwoDayClean_Reuse")

# solver settings
m.Params.Threads = 0
m.Params.MIPFocus = 1
m.Params.Heuristics = 0.2
m.Params.Cuts = 2
m.Params.Presolve = 2
m.Params.NodeMethod = 1
m.Params.OutputFlag = 1

# depot open decision
y = m.addVars(J, vtype=GRB.BINARY, name="y")

# total fleet size per mode (max over both days)
n = m.addVars(T, vtype=GRB.INTEGER, lb=0, name="n")

# route assignment variables
z = {}
for day in T_days:
    z_day = m.addVars(IJTS[day], vtype=GRB.BINARY, name=f"z_{day}")
    for (i, j, t) in IJTS[day]:
        z[i, j, t, day] = z_day[i, j, t]

# vehicles per depot, mode and day
n_jt = {}
for day in T_days:
    n_jt_day = m.addVars(J, T, vtype=GRB.INTEGER, lb=0, ub=50, name=f"n_jt_{day}")
    for j in J:
        for t in T:
            n_jt[j, t, day] = n_jt_day[j, t]

# total working hours per mode and day
g = {}
for day in T_days:
    g_day = m.addVars(T, lb=0, name=f"g_{day}")
    for t in T:
        g[t, day] = g_day[t]

m.update()
print(f"[ok] total variables: {m.NumVars}")

# =============================================================================
# 6) objective function
# =============================================================================
print("[step 6] setting objective function")

depot_fixed = 2 * quicksum(f_j[j] * y[j] for j in J)
vehicle_fixed = 2 * quicksum(c_t_s[t] * n[t] for t in T)

storage_cost = quicksum(
    c_s * D[i] * z[i, j, t, day]
    for day in T_days
    for (i, j, t) in IJTS[day]
)

labor_cost = quicksum(
    (c_t[t] + w) * g[t, day]
    for t in T
    for day in T_days
)

m.setObjective(
    depot_fixed + vehicle_fixed + storage_cost + labor_cost,
    GRB.MINIMIZE
)

# =============================================================================
# 7) constraints
# =============================================================================
print("[step 7] adding constraints")

constr_count = 0

def report(name, n):
    global constr_count
    constr_count += n
    print(f"  [ok] {name:<40s}: {n:>8,d}   (total: {constr_count:>8,d})")

# ------------------------------------------------------------
# 1. each route must be served exactly once on its service day
# ------------------------------------------------------------
cnt = 0
for day in T_days:
    I = I_by_day[day]
    ijts = IJTS[day]
    m.addConstrs(
        (quicksum(z[i2, j2, t2, day]
                  for (i2, j2, t2) in ijts if i2 == i) == 1
         for i in I),
        name=f"{day}_route_once"
    )
    cnt += len(I)
report("route assignment (exactly once)", cnt)

# ------------------------------------------------------------
# 2. routes can only be assigned to open depots
# ------------------------------------------------------------
cnt = 0
for day in T_days:
    ijts = IJTS[day]
    m.addConstrs(
        (z[i, j, t, day] <= y[j] for (i, j, t) in ijts),
        name=f"{day}_depot_open"
    )
    cnt += len(ijts)
report("depot must be open", cnt)

# ------------------------------------------------------------
# 3. depot demand capacity constraint
# ------------------------------------------------------------
cnt = 0
for day in T_days:
    ijts = IJTS[day]
    for j in J:
        m.addConstr(
            quicksum(
                D[i2] * z[i2, j2, t2, day]
                for (i2, j2, t2) in ijts if j2 == j
            ) <= C_d * y[j],
            name=f"{day}_depot_capacity_{j}"
        )
        cnt += 1
report("depot demand capacity", cnt)

# ------------------------------------------------------------
# 4. per-route time feasibility by mode
# ------------------------------------------------------------
cnt = 0
for day in T_days:
    for (i, j, t) in IJTS[day]:
        travel = (d[(j, i)] + d[(i, j)]) / v_t[t]
        service = S[(i, j, t)]
        m.addConstr(
            z[i, j, t, day] * (travel + service) <= R_t[t],
            name=f"{day}_route_time_{i}_{j}_{t}"
        )
        cnt += 1
report("per-route time feasibility", cnt)

# ------------------------------------------------------------
# 5. depot time capacity per mode (vehicle reuse)
# ------------------------------------------------------------
cnt = 0
for day in T_days:
    ijts = IJTS[day]
    for j in J:
        for t in T:
            m.addConstr(
                quicksum(
                    z[i2, j2, t2, day] *
                    ((d[(j2, i2)] + d[(i2, j2)]) / v_t[t2] + S[(i2, j2, t2)])
                    for (i2, j2, t2) in ijts
                    if j2 == j and t2 == t
                ) <= n_jt[j, t, day] * R_t[t],
                name=f"{day}_depot_time_{j}_{t}"
            )
            cnt += 1
report("depot time capacity (reuse)", cnt)

# ------------------------------------------------------------
# 6. fleet size is max over both days
# ------------------------------------------------------------
cnt = 0
for t in T:
    m.addConstr(
        n[t] >= quicksum(n_jt[j, t, "X"] for j in J),
        name=f"fleet_X_{t}"
    )
    m.addConstr(
        n[t] >= quicksum(n_jt[j, t, "Y"] for j in J),
        name=f"fleet_Y_{t}"
    )
    cnt += 2
report("fleet sizing (max over days)", cnt)

# ------------------------------------------------------------
# 7. global working time accounting
# ------------------------------------------------------------
cnt = 0
for day in T_days:
    ijts = IJTS[day]
    for t in T:
        travel = quicksum(
            z[i2, j2, t2, day] *
            (d[(j2, i2)] + d[(i2, j2)]) / v_t[t2]
            for (i2, j2, t2) in ijts if t2 == t
        )
        service = quicksum(
            z[i2, j2, t2, day] * S[(i2, j2, t2)]
            for (i2, j2, t2) in ijts if t2 == t
        )
        m.addConstr(
            g[t, day] >= travel + service,
            name=f"{day}_time_accounting_{t}"
        )
        cnt += 1
report("global time accounting", cnt)

print(f"[ok] total constraints in model: {m.NumConstrs:,}")

# =============================================================================
# 8) solve model
# =============================================================================
print("[step 8] solving model")

m.Params.MIPGap = 0.01
m.Params.TimeLimit = 3600
m.optimize()

print(f"[status] solver status: {m.Status}")

if m.Status == GRB.INFEASIBLE:
    print("[error] model infeasible, writing iis")
    m.computeIIS()
    m.write("model_iis.ilp")
    exit()

# =============================================================================
# 9) results and reporting
# =============================================================================
print("[step 9] processing results")

if m.SolCount == 0:
    print("[warn] no feasible solution found")
    exit()

print(f"[ok] objective value (2 days): €{m.ObjVal:,.2f}")
print(f"[ok] optimality gap: {m.MIPGap:.2%}")

# cost breakdown
vehicle_fixed_cost_2days = 2 * sum(c_t_s[t] * n[t].X for t in T)
depot_cost_2days = 2 * sum(f_j[j] * y[j].X for j in J)
labor_electricity_cost = sum((c_t[t] + w) * g[t, day].X for t in T for day in T_days)
storage_handling_cost = sum(
    c_s * D[i] * z[i, j, t, day].X
    for day in T_days
    for (i, j, t) in IJTS[day]
)

print("[summary] cost breakdown (2 days)")
print(f"  depots: €{depot_cost_2days:,.2f}")
print(f"  vehicles: €{vehicle_fixed_cost_2days:,.2f}")
print(f"  storage/handling: €{storage_handling_cost:,.2f}")
print(f"  labor + electricity: €{labor_electricity_cost:,.2f}")

# ============================================================
# 9) FULL RESULTS & EXCEL EXPORT
# ============================================================
if m.SolCount > 0:
    print(f"\n{'='*60}")
    print(f"Objective value: €{m.ObjVal:,.2f}")
    print(f"Optimality gap: {m.MIPGap:.2%}")
    print(f"{'='*60}")

    model_summary = [{
        "objective_value_2days": round(m.ObjVal, 2),
        "objective_value_daily": round(m.ObjVal / 2, 2),
        "objective_value_annual": round(m.ObjVal * (365 / 2), 2),
        "optimality_gap": round(m.MIPGap, 4),
        "status": m.Status,
        "num_routes_total": len(I_all),
        "num_routes_X": len(I_by_day["X"]),
        "num_routes_Y": len(I_by_day["Y"]),
        "num_depots": len(J),
        "num_modes": len(T),
        "num_variables": m.NumVars,
        "num_constraints": m.NumConstrs,
    }]

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

    # Day summary
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
        print(f"[{day}]")
        print(f" Routes: {total_routes}")
        print(f" Demand: {total_demand} bags")
        print(f" Work hours: {total_work:.2f}")
        print(f" Depots: {depots_open}")

    # Mode usage
    mode_rows = []
    for t in T:
        for day in T_days:
            work_time = g[t, day].X
            routes_count = sum(1 for (i, j, t2) in IJTS[day] if t2 == t and z[i, j, t, day].X > 0.5)
            vehicles_deployed = sum(n_jt[j, t, day].X for j in J)
            max_capacity = vehicles_deployed * R_t[t] if vehicles_deployed > 0 else 0.0
            mode_rows.append({
                "day": day,
                "mode": t,
                "vehicles_total": int(round(n[t].X)),
                "vehicles_deployed": round(vehicles_deployed, 1),
                "num_routes": routes_count,
                "work_time_hours": round(work_time, 2),
                "max_capacity_hours": round(max_capacity, 2),
                "utilization_pct": round(100 * work_time / max_capacity, 1) if max_capacity > 0 else 0.0,
            })

    # Depot usage
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

    # Depot-Mode matrix (NOW SHOWS ACTUAL VEHICLES!)
    depot_mode_rows = []
    for day in T_days:
        for j in J:
            for t in T:
                vehicles_at_depot = round(n_jt[j, t, day].X, 1)
                depot_mode_rows.append({
                    "day": day,
                    "depot": j,
                    "mode": t,
                    "vehicles_allocated": vehicles_at_depot,
                    "depot_open": int(round(y[j].X)),
                })

    # Route details
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

    # WRITE FULL EXCEL
    out_path = os.path.join(SCRIPT_DIR, "solution_Reuse.xlsx")
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        pd.DataFrame(model_summary).to_excel(writer, sheet_name="ModelSummary", index=False)
        pd.DataFrame(cost_rows).to_excel(writer, sheet_name="CostBreakdown", index=False)
        pd.DataFrame(day_summary).to_excel(writer, sheet_name="DaySummary", index=False)
        pd.DataFrame(mode_rows).to_excel(writer, sheet_name="ModeUsage", index=False)
        pd.DataFrame(depot_rows).to_excel(writer, sheet_name="DepotUsage", index=False)
        pd.DataFrame(depot_mode_rows).to_excel(writer, sheet_name="DepotModeMatrix", index=False)
        pd.DataFrame(route_rows).to_excel(writer, sheet_name="Routes", index=False)

    print(f"[ok] FULL solution written to: {out_path}")
    print("\n" + "=" * 80)
    print("OBJECTIVE BREAKDOWN:")
    print("=" * 80)
    print(f"Vehicles (2 days, fixed): €{vehicle_fixed_cost_2days:,.2f}")
    print(f"Depots (2 days, fixed): €{depot_cost_2days:,.2f}")
    print(f"Storage/Handling: €{storage_handling_cost:,.2f}")
    print(f"Labor + Electricity: €{labor_electricity_cost:,.2f}")
    print(f" " + "-" * 20)
    print(f"TOTAL (2 days): €{m.ObjVal:,.2f}")
    print(f"Daily average: €{m.ObjVal / 2:,.2f}")
    print(f"Annual: €{m.ObjVal * (365 / 2):,.2f}")
    print("=" * 80)
else:
    print("[warn] no feasible solution")

print("Script is finished")
