import os
import pickle
import pandas as pd
from collections import defaultdict
from gurobipy import Model, GRB, quicksum

print("=" * 80)
print("starting depot routing model - TWO-DAY CLEAN FLEET WITH ROUTE XY SPLIT")
print("=" * 80)

# ============================================================
# 1) load data
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_NAME = "model_inputUSXY.pkl"

print(f"[step 1] loading data from {PKL_NAME}")

with open(os.path.join(SCRIPT_DIR, PKL_NAME), "rb") as f:
    model_data = pickle.load(f)


# sets
I_all = list(model_data["sets"]["I"])  # all routes / delivery clusters
J = list(model_data["sets"]["J"])  # depots (SHARED)
T = list(model_data["sets"]["T"])  # transport modes

# parameters
D = model_data["parameters"]["D"]  # demand per route
d = model_data["parameters"]["d"]  # distance matrix
S = model_data["parameters"]["service_time"]  # service time per (route, depot, mode)

print(f"[ok] |I_all|={len(I_all)}, |J|={len(J)}, |T|={len(T)}")
print(f"[ok] |distance arcs|={len(d)}, |service entries|={len(S)}")

# ============================================================
# 1B) SPLIT ROUTES INTO TWO DAY TYPES (BY EXPLICIT ROUTE XY CODE)
# ============================================================

print("[step 1B] splitting routes by XY code (X = Tue/Thu, Y = Wed/Fri)")

T_days = ["TueThur", "WedFri"]

# Load route XY codes from model_data
xy_codes = model_data["parameters"].get("xy", {})

if not xy_codes:
    print("[error] XY codes not found in parameters!")
    raise ValueError("Cannot split routes: XY codes missing from pickle")

I_by_day = {
    "TueThur": [],  # Type X routes (Tuesday/Thursday)
    "WedFri": []    # Type Y routes (Wednesday/Friday)
}

# Split all routes by their XY code (all routes have valid X or Y)
for i in I_all:
    xy_code = xy_codes.get(i)
    
    if xy_code == "X":
        I_by_day["TueThur"].append(i)
    elif xy_code == "Y":
        I_by_day["WedFri"].append(i)
    else:
        print(f"[error] route {i} has invalid XY code: {xy_code}")

print(f"[ok] Tue/Thu routes (XY=X): {len(I_by_day['TueThur'])}")
print(f"[ok] Wed/Fri routes (XY=Y): {len(I_by_day['WedFri'])}")

# ============================================================
# 2) parameters
# ============================================================

print("[step 2] building parameters")

c_s = 0.42  # storage / handling cost per delivery bag (per day)
f_j = {j: 50 for j in J}  # fixed depot opening costs
C_d = 10000  # depot capacity


# vehicle capacities (bags)
Q_t = {
    "FootBike": 4,
    "Car": 25,
    "EBikeCart": 6,
    "Moped": 4
}

# vehicle fixed costs per unit
c_t_s = {
    "FootBike": 0.52,
    "Car": 95.53,
    "EBikeCart": 14.58,
    "Moped": 18.54
}

# variable hourly vehicle costs
c_t = {
    "FootBike": 0.0,
    "Car": 1.07,
    "EBikeCart": 0.03,
    "Moped": 0.22
}

# average speeds (km/h)
v_t = {
    "FootBike": 13,
    "Car": 35,
    "EBikeCart": 17.5,
    "Moped": 22.5
}

# maximum daily uptime (hours)
R_t = {
    "FootBike": 3.0,
    "Car": 5.5,
    "EBikeCart": 3.5,
    "Moped": 4.5
}

w = 23.30  # labor cost per hour

# ============================================================
# 3) candidate depots
# ============================================================

print("[step 3] selecting candidate depots")

N_DEPOTS = 20
J_cand = {}
missing = 0

for i in I_all:
    cand = [(j, d[(i, j)]) for j in J if (i, j) in d]
    cand.sort(key=lambda x: x[1])
    if not cand:
        missing += 1
    J_cand[i] = [j for (j, _) in cand[:N_DEPOTS]]

print(f"[ok] candidate depots per route = {N_DEPOTS}")
print(f"[check] routes without depot distances: {missing}")

# identify routes that have no candidate depots at all
bad_routes = [i for i in I_all if len(J_cand[i]) == 0]

if bad_routes:
    print(f"[warn] removing {len(bad_routes)} unreachable routes")
    I_all = [i for i in I_all if i not in bad_routes]
    for day in T_days:
        I_by_day[day] = [i for i in I_by_day[day] if i not in bad_routes]
    D = {i: D[i] for i in I_all}
    J_cand = {i: J_cand[i] for i in I_all}

print(f"[ok] remaining routes after cleanup: {len(I_all)}")

# ============================================================
# 4) sparse arcs
# ============================================================

print("[step 4] building sparse arc set (per day)")

A_by_day = {}
A_dr_by_day = {}
A_rd_by_day = {}
A_rr_by_day = {}
OUT_by_day = {}
IN_by_day = {}

NEIGH_ROUTES = 15

for day in T_days:
    I = I_by_day[day]
    A_dr, A_rd, A_rr = [], [], []
    
    # depot ↔ route arcs
    for i in I:
        for j in J_cand[i]:
            if (j, i) in d:
                A_dr.append((j, i))
            if (i, j) in d:
                A_rd.append((i, j))
    
    # route → route arcs (k-nearest neighbors)
    rr_by_u = defaultdict(list)
    for (u, v), dist in d.items():
        if u in I and v in I and u != v:
            rr_by_u[u].append((v, dist))
    
    for u in I:
        rr_by_u[u].sort(key=lambda x: x[1])
        for v, _ in rr_by_u[u][:NEIGH_ROUTES]:
            A_rr.append((u, v))
    
    A = list(dict.fromkeys(A_dr + A_rr + A_rd))
    
    OUT = defaultdict(list)
    IN = defaultdict(list)
    
    for (u, v) in A:
        OUT[u].append(v)
        IN[v].append(u)
    
    A_by_day[day] = A
    A_dr_by_day[day] = A_dr
    A_rd_by_day[day] = A_rd
    A_rr_by_day[day] = A_rr
    OUT_by_day[day] = OUT
    IN_by_day[day] = IN
    
    print(f"[ok] {day}: |A_dr|={len(A_dr)}, |A_rr|={len(A_rr)}, |A_rd|={len(A_rd)}, |A|={len(A)}")

# ============================================================
# 5) travel times
# ============================================================

print("[step 5] precomputing travel times")

L_by_day = {}

for day in T_days:
    A = A_by_day[day]
    L_by_day[day] = {(u, v, t): d[(u, v)] / v_t[t] for (u, v) in A for t in T}

# ============================================================
# 6) model + variables
# ============================================================

print("[step 6] creating model and variables")

m = Model("DepotRoutingCleanFleet_TwoDay_XYSplit")

# Shared variables (across both days)
y = m.addVars(J, vtype=GRB.BINARY, name="y")

# CLEAN FLEET: single pool per mode, sized for peak day, owned both days
n = m.addVars(T, vtype=GRB.INTEGER, lb=0, name="n")

# n_jt[j,t,day]: vehicles of mode t stationed at depot j on given day
n_jt = {}

for day in T_days:
    n_jt_day = m.addVars(J, T, vtype=GRB.INTEGER, lb=0, name=f"n_jt_{day}")
    for j in J:
        for t in T:
            n_jt[j, t, day] = n_jt_day[j, t]

# Per-day variables
x = {}

for day in T_days:
    A = A_by_day[day]
    x_day = m.addVars(A, T, vtype=GRB.BINARY, lb=0, name=f"x_{day}")
    for (u, v) in A:
        for t in T:
            x[u, v, t, day] = x_day[u, v, t]

r = {}

for day in T_days:
    I = I_by_day[day]
    r_day = m.addVars(I, T, vtype=GRB.BINARY, name=f"r_{day}")
    for i in I:
        for t in T:
            r[i, t, day] = r_day[i, t]

z = {}

for day in T_days:
    I = I_by_day[day]
    z_day = m.addVars([(i, j) for i in I for j in J_cand[i]], vtype=GRB.BINARY, name=f"z_{day}")
    for i in I:
        for j in J_cand[i]:
            z[i, j, day] = z_day[i, j]

g = {}

for day in T_days:
    g_day = m.addVars(T, lb=0, name=f"g_{day}")
    for t in T:
        g[t, day] = g_day[t]

m.update()

# ============================================================
# 7) objective
# ============================================================

print("[step 7] setting objective")

# Depot opening
depot_cost = 2 * quicksum(f_j[j] * y[j] for j in J)

# Vehicle fixed cost
vehicle_startup_cost = 2 * quicksum(c_t_s[t] * n[t] for t in T)

# Operational costs
operational_cost = 0

for day in T_days:
    I = I_by_day[day]

    for i in I:
        for j in J_cand[i]:
            operational_cost += c_s * D[i] * z[i, j, day]

    for t in T:
        operational_cost += (c_t[t] + w) * g[t, day]

m.setObjective(
    depot_cost + vehicle_startup_cost + operational_cost,
    GRB.MINIMIZE
)

# ============================================================
# 8) constraints
# ============================================================

print("[step 8] adding constraints")

constr_count = 0

def report(name, n):
    global constr_count
    constr_count += n
    print(f" [ok] {name:<50s}: {n:>8,d} (total: {constr_count:>8,d})")

for day in T_days:
    I = I_by_day[day]
    A = A_by_day[day]
    A_dr = A_dr_by_day[day]
    A_rd = A_rd_by_day[day]
    A_rr = A_rr_by_day[day]
    OUT = OUT_by_day[day]
    IN = IN_by_day[day]
    L = L_by_day[day]
    
    for i in I:
        m.addConstr(quicksum(x[u, i, t, day] for u in IN[i] for t in T) == 1)
    report(f"{day}: 8.1 assignment (visit once)", len(I))
    
    for i in I:
        m.addConstr(quicksum(r[i, t, day] for t in T) == 1)
    report(f"{day}: 8.2 route–mode assignment", len(I))
    
    for i in I:
        for t in T:
            m.addConstr(quicksum(x[u, i, t, day] for u in IN[i]) <= r[i, t, day])
    report(f"{day}: 8.3 flow–mode linking", len(I) * len(T))
    
    for t in T:
        for i in I:
            m.addConstr(
                quicksum(x[u, i, t, day] for u in IN[i]) ==
                quicksum(x[i, v, t, day] for v in OUT[i])
            )
    report(f"{day}: 8.4 flow conservation", len(I) * len(T))
    
    for i in I:
        m.addConstr(quicksum(z[i, j, day] for j in J_cand[i]) == 1)
    report(f"{day}: 8.5 route–depot assignment", len(I))
    
    for i in I:
        for j in J_cand[i]:
            m.addConstr(z[i, j, day] <= y[j])
    report(f"{day}: 8.6 z <= y", sum(len(J_cand[i]) for i in I))
    
    for j in J:
        m.addConstr(
            quicksum(D[i] * z[i, j, day] for i in I if (i, j) in z)
            <= C_d * y[j]
        )
    report(f"{day}: 8.7 depot capacity", len(J))
    
    # ============================================================
    # 8.8 ROUTING CONSISTENCY & TOUR RESTRICTIONS
    # ============================================================

    # ------------------------------------------------------------
    # 8.8a depot arc consistency
    # depot → route and route → depot arcs only allowed
    # if the route is assigned to that depot
    # ------------------------------------------------------------
    for (j, i) in A_dr:
        for t in T:
            m.addConstr(x[j, i, t, day] <= z[i, j, day])

    for (i, j) in A_rd:
        for t in T:
            m.addConstr(x[i, j, t, day] <= z[i, j, day])

    report(
        f"{day}: 8.8a depot arc consistency",
        len(A_dr) * len(T) + len(A_rd) * len(T)
    )

    # ------------------------------------------------------------
    # 8.8b route–route arcs must stay within the same depot
    # prevents cross-depot tours
    # ------------------------------------------------------------
    for (u, v) in A_rr:
        for t in T:
            m.addConstr(
                x[u, v, t, day] <=
                quicksum(
                    z[u, j, day] * z[v, j, day]
                    for j in J_cand[u]
                    if j in J_cand[v]
                )
            )

    report(
        f"{day}: 8.8b route–route depot consistency",
        len(A_rr) * len(T)
    )

    # ------------------------------------------------------------
    # 8.8c forbid route–route chaining for non-car modes
    # only cars are allowed to chain routes
    # ------------------------------------------------------------
    cnt_noncar = 0
    for (u, v) in A_rr:
        for t in T:
            if t != "Car":
                m.addConstr(x[u, v, t, day] == 0)
                cnt_noncar += 1

    report(
        f"{day}: 8.8c rr forbidden for non-car modes",
        cnt_noncar
    )

    # ------------------------------------------------------------
    # 8.8d car capacity feasibility for 2-route tours
    # only allow car chaining if combined demand fits capacity
    # ------------------------------------------------------------
    cnt_cap = 0
    Qcar = Q_t["Car"]

    for (u, v) in A_rr:
        if D[u] + D[v] > Qcar:
            m.addConstr(x[u, v, "Car", day] == 0)
            cnt_cap += 1

    report(
        f"{day}: 8.8d car rr capacity cutoff",
        cnt_cap
    )

    # ------------------------------------------------------------
    # 8.8e limit car tours to max 2 routes
    # a route may not have both an incoming and outgoing
    # car route–route arc
    # ------------------------------------------------------------
    cnt_len = 0
    for i in I:
        m.addConstr(
            quicksum(x[u, i, "Car", day] for (u, i2) in A_rr if i2 == i) +
            quicksum(x[i, v, "Car", day] for (i2, v) in A_rr if i2 == i)
            <= 1
        )
        cnt_len += 1

    report(
        f"{day}: 8.8e car tour length ≤ 2 routes",
        cnt_len
    )





    for t in T:
        for j in J:
            m.addConstr(
                quicksum(x[j, i, t, day] for (j2, i) in A_dr if j2 == j) ==
                quicksum(x[i, j, t, day] for (i, j2) in A_rd if j2 == j)
            )
    report(f"{day}: 8.9 depot start/end balance", len(J) * len(T))

    for t in T:
        m.addConstr(
            g[t, day] >=
            quicksum(x[j, i, t, day] * L[(j, i, t)] for (j, i) in A_dr) +
            quicksum(x[u, v, t, day] * L[(u, v, t)] for (u, v) in A_rr) +
            quicksum(x[i, j, t, day] * L[(i, j, t)] for (i, j) in A_rd) +
            quicksum(
                r[i, t, day] * z[i, j, day] * S.get((i, j, t), 0.0)
                for i in I for j in J_cand[i]
            )
        )
    report(f"{day}: 8.10 global time accounting", len(T))

for day in T_days:
    A_dr = A_dr_by_day[day]
    for j in J:
        for t in T:
            m.addConstr(
                n_jt[j, t, day] == quicksum(x[j, i, t, day] for (j2, i) in A_dr if j2 == j)
            )

report("FLEET: 8.11 n_jt = start counts per depot per mode", 2 * len(J) * len(T))

for t in T:
    n_sum_tue = quicksum(n_jt[j, t, "TueThur"] for j in J)
    n_sum_wed = quicksum(n_jt[j, t, "WedFri"] for j in J)
    
    m.addConstr(n[t] >= n_sum_tue)
    m.addConstr(n[t] >= n_sum_wed)

report("FLEET: 8.12a total n[t] = max(Tue, Wed) per mode", 2 * len(T))

for day in T_days:
    I = I_by_day[day]
    for t in T:
        routes_using_mode = quicksum(r[i, t, day] for i in I)
        m.addConstr(routes_using_mode <= n[t])

report("FLEET: 8.12b routes <= fleet per day", 2 * len(T))

for day in T_days:
    for t in T:
        m.addConstr(g[t, day] <= n[t] * R_t[t])

report("FLEET: 8.12c uptime constraint per day (RESETS DAILY)", 2 * len(T))

print(f"[ok] total constraints added: {constr_count:,}")

# ============================================================
# 9) solve
# ============================================================

print("[step 9] optimizing")

m.Params.OutputFlag = 1
m.Params.MIPGap = 0.01
m.Params.Method = 3
m.Params.TimeLimit = 20000

m.optimize()

sol_path = os.path.join(SCRIPT_DIR, "Solution_TwoDay.sol")
m.write(sol_path)

print(f"[ok] solution written to: {sol_path}")
print("[status]", m.Status)
print(f"[info] best gap achieved = {m.MIPGap:.4%}")

if m.Status == GRB.INFEASIBLE:
    print("[diag] model infeasible: computing iis")
    m.computeIIS()
    m.write("model_iis.ilp")

# ============================================================
# 10) EXPORT RESULTS
# ============================================================

print("[step 10] exporting results (direct from model variables)")

if m.SolCount > 0:
    print(f"[ok] best objective value = {m.ObjVal:.4f}")
    print(f"[ok] optimality gap       = {m.MIPGap:.4%}")
    print(f"[ok] solver status        = {m.Status}")

    # ========== 1) MODEL SUMMARY ==========
    model_summary = [{
        "objective_value_2days": round(m.ObjVal, 2),
        "objective_value_daily": round(m.ObjVal / 2, 2),
        "objective_value_annual": round(m.ObjVal * (365 / 2), 2),
        "optimality_gap": m.MIPGap,
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
    storage_handling_cost = sum(c_s * D[i] * z[i, j, day].X 
                                for day in T_days 
                                for i in I_by_day[day] 
                                for j in J_cand[i] 
                                if (i, j, day) in z)
    
    cost_rows = [{
        "cost_depots_2days": round(depot_cost_2days, 2),
        "cost_vehicles_fixed_2days": round(vehicle_fixed_cost_2days, 2),
        "cost_storage_handling": round(storage_handling_cost, 2),
        "cost_labor_electricity": round(labor_electricity_cost, 2),
        "total_cost_2days": round(m.ObjVal, 2),
        "cost_daily_average": round(m.ObjVal / 2, 2),
        "cost_annual": round(m.ObjVal * (365 / 2), 2),
    }]

    # ========== 3) DEPOT USAGE (per day) ==========
    depot_rows = []
    for j in J:
        for day in T_days:
            assigned_routes = [i for i in I_by_day[day] if (i, j, day) in z and z[i, j, day].X > 0.5]
            demand = sum(D[i] for i in assigned_routes)
            
            depot_rows.append({
                "day": day,
                "depot": j,
                "open": int(round(y[j].X)),
                "num_routes": len(assigned_routes),
                "total_demand": demand,
            })

    # ========== 4) MODE USAGE (per day) ==========
    mode_rows = []
    for t in T:
        for day in T_days:
            work_time = g[t, day].X
            routes_count = sum(1 for i in I_by_day[day] if (i, t, day) in r and r[i, t, day].X > 0.5)
            
            # vehicles_deployed = count of depots that deployed >= 1 vehicle of mode t on this day
            vehicles_deployed = sum(int(round(n_jt[j, t, day].X)) for j in J)

            # max_capacity = vehicles_deployed × R_t[t] (uptime per vehicle per day)
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

    # ========== 5) DEPOT-MODE MATRIX  ==========
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

    # ========== 6) DAY SUMMARY ==========
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

    # ========== 7) ROUTE-LEVEL DETAILS ==========
    route_rows = []
    for day in T_days:
        I = I_by_day[day]
        for i in I:
            depot = None
            for j in J_cand[i]:
                if (i, j, day) in z and z[i, j, day].X > 0.5:
                    depot = j
                    break
            if depot is None:
                continue
            
            mode = None
            for t in T:
                if (i, t, day) in r and r[i, t, day].X > 0.5:
                    mode = t
                    break
            if mode is None:
                continue
            
            t_srv = S.get((i, depot, mode), 0.0)
            t_out = L_by_day[day].get((depot, i, mode), 0.0)
            t_back = L_by_day[day].get((i, depot, mode), 0.0)
            time_total = t_out + t_srv + t_back
            
            cost_storage = c_s * D[i]
            cost_labor = (w + c_t[mode]) * time_total
            cost_route_total = cost_storage + cost_labor
            
            route_rows.append({
                "day": day,
                "route": i,
                "depot": depot,
                "mode": mode,
                "demand": D[i],
                "service_time_hours": round(t_srv, 2),
                "travel_out_hours": round(t_out, 2),
                "travel_back_hours": round(t_back, 2),
                "total_time_hours": round(time_total, 2),
                "cost_storage_handling": round(cost_storage, 2),
                "cost_labor": round(cost_labor, 2),
                "cost_route_total": round(cost_route_total, 2),
            })

    # ========== WRITE TO EXCEL ==========
    out_path = os.path.join(SCRIPT_DIR, "solution_twoday.xlsx")

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
    print(f"Vehicles (2 days, fixed):  €{vehicle_fixed_cost_2days:,.2f}")
    print(f"Depots (2 days, fixed):    €{depot_cost_2days:,.2f}")
    print(f"Storage/Handling (c_s):    €{storage_handling_cost:,.2f}")
    print(f"Labor + Electricity:       €{labor_electricity_cost:,.2f}")
    print(f"                           " + "-" * 20)
    print(f"TOTAL (2 days):            €{m.ObjVal:,.2f}")
    print(f"Daily average:             €{m.ObjVal/2:,.2f}")
    print(f"Annual (scaled):           €{m.ObjVal * (365/2):,.2f}")
    print("=" * 80)

else:
    print("[warn] no feasible solution — nothing exported")

print("[✓] export complete!")
