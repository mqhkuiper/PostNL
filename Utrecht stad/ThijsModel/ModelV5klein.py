'''
depot + routing milp
aggregated vehicles with route–mode assignment

this model:
 - uses sparse routing arcs to reduce size
 - allows depot-dependent service times
 - sizes the fleet per transport mode (aggregated)
 - assigns each route to exactly one depot and one mode

'''

import os
import pickle
import pandas as pd
from collections import defaultdict
from gurobipy import Model, GRB, quicksum

print("=" * 80)
print("starting depot routing model aggregated")
print("=" * 80)

# ============================================================
# 1) load data
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_NAME = "model_input_US.pkl"

print(f"[step 1] loading data from {PKL_NAME}")

with open(os.path.join(SCRIPT_DIR, PKL_NAME), "rb") as f:
    model_data = pickle.load(f)

I = list(model_data["sets"]["I"])
J = list(model_data["sets"]["J"])
T = list(model_data["sets"]["T"])

D = model_data["parameters"]["D"]
d = model_data["parameters"]["d"]
S = model_data["parameters"]["service_time"]

print(f"[ok] |I|={len(I)}, |J|={len(J)}, |T|={len(T)}")
print(f"[ok] |distance arcs|={len(d)}, |service entries|={len(S)}")

# ============================================================
# 2) parameters --> zijn nog niet kloppend
# ============================================================

print("[step 2] building parameters")

# depot related
c_s = 0.75                       # storage cost per delivery bag per day
f_j = {j: 0.0 for j in J}        # opening costs not yet determined
C_d = 10000                      # non-binding depot capacity

# vehicle fixed daily costs
c_t_s = {
    "FootBike": 0.52,            # bicycle allowance
    "Car": 66.73,                # storage + lease
    "EBikeCart": 16.02,          # storage + lease
    "Moped": 12.54               # storage only (lease unknown)
}

# vehicle operational (hourly) costs
c_t = {
    "FootBike": 0.0,
    "Car": 1.07,
    "EBikeCart": 0.03,
    "Moped": 0.22
}

# capacities (delivery bags)
Q_t = {
    "FootBike": 3,
    "Car": 25,
    "EBikeCart": 5,
    "Moped": 3
}

# velocities (km/hour)
v_t = {
    "FootBike": 15,
    "Car": 40,
    "EBikeCart": 17.5,
    "Moped": 30
}

# maximum daily uptime (hours)
R_t = {
    "FootBike": 3.0,
    "Car": 5.5,
    "EBikeCart": 3.5,
    "Moped": 4.5
}

# other parameters
M = 10000
w = 23.30

# ============================================================
# 3) candidate depots
# ============================================================

print("[step 3] selecting candidate depots")

N_DEPOTS = 20
J_cand = {}
missing = 0

for i in I:
    cand = [(j, d[(i, j)]) for j in J if (i, j) in d]
    cand.sort(key=lambda x: x[1])
    if not cand:
        missing += 1
    J_cand[i] = [j for (j, _) in cand[:N_DEPOTS]]

print(f"[ok] candidate depots per route = {N_DEPOTS}")
print(f"[check] routes without depot distances: {missing}")

bad_routes = [i for i in I if len(J_cand[i]) == 0]

if bad_routes:
    print(f"[warn] removing {len(bad_routes)} unreachable routes:")
    for i in bad_routes:
        print("   ", i)

    I = [i for i in I if i not in bad_routes]
    D = {i: D[i] for i in I}
    J_cand = {i: J_cand[i] for i in I}

print(f"[ok] remaining routes after cleanup: {len(I)}")

# ============================================================
# 4) sparse arcs
# ============================================================

print("[step 4] building sparse arc set")

A_dr, A_rd, A_rr = [], [], []

for i in I:
    for j in J_cand[i]:
        if (j, i) in d:
            A_dr.append((j, i))
        if (i, j) in d:
            A_rd.append((i, j))

NEIGH_ROUTES = 15
rr_by_u = defaultdict(list)

for (u, v), dist in d.items():
    if u in I and v in I and u != v:
        rr_by_u[u].append((v, dist))

for u in I:
    rr_by_u[u].sort(key=lambda x: x[1])
    for v, _ in rr_by_u[u][:NEIGH_ROUTES]:
        A_rr.append((u, v))

A = list(dict.fromkeys(A_dr + A_rr + A_rd))

print(f"[ok] depot->route arcs: {len(A_dr)}")
print(f"[ok] route->route arcs: {len(A_rr)}")
print(f"[ok] route->depot arcs: {len(A_rd)}")
print(f"[ok] total arcs |A| = {len(A)}")

OUT = defaultdict(list)
IN = defaultdict(list)
for (u, v) in A:
    OUT[u].append(v)
    IN[v].append(u)

# ============================================================
# 5) travel times
# ============================================================

print("[step 5] precomputing travel times")

L = {(u, v, t): d[(u, v)] / v_t[t] for (u, v) in A for t in T}

print(f"[ok] travel-time entries: {len(L)}")

# ============================================================
# 6) model + variables
# ============================================================

print("[step 6] creating model and variables")

m = Model("DepotRoutingAggregated")

x = m.addVars(A, T, vtype=GRB.INTEGER, lb=0, name="x")
r = m.addVars(I, T, vtype=GRB.BINARY, name="r")
y = m.addVars(J, vtype=GRB.BINARY, name="y")
z = m.addVars([(i, j) for i in I for j in J_cand[i]], vtype=GRB.BINARY, name="z")
g = m.addVars(T, lb=0, name="g")
n = m.addVars(T, vtype=GRB.INTEGER, lb=0, name="n")

print("[ok] variables created")

# ============================================================
# 7) objective
# ============================================================

print("[step 7] setting objective")

m.setObjective(
    quicksum(f_j[j] * y[j] for j in J)
    + 0.56 * quicksum(c_s * D[i] * z[i, j] for (i, j) in z)
    + 0.5 * quicksum(c_t_s[t] * x[j, i, t] for (j, i) in A_dr for t in T)
    + quicksum((c_t[t] + w) * g[t] for t in T),
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
    print(f"  [ok] {name:<42s}: {n:>8,d}   (total: {constr_count:>8,d})")

# each route must be visited exactly once (one incoming arc across all modes)
cnt = 0
for i in I:
    m.addConstr(
        quicksum(x[u, i, t] for u in IN[i] for t in T) == 1
    )
    cnt += 1
report("assignment (visit once)", cnt)

# each route chooses exactly one transport mode
cnt = 0
for i in I:
    m.addConstr(
        quicksum(r[i, t] for t in T) == 1
    )
    cnt += 1
report("route–mode assignment", cnt)

# flow into a route using mode t is only allowed if that mode is selected
cnt = 0
for i in I:
    for t in T:
        m.addConstr(
            quicksum(x[u, i, t] for u in IN[i]) <= r[i, t]
        )
        cnt += 1
report("flow–mode linking", cnt)

# flow conservation per route and per mode
# ensures that if a vehicle enters a route, it also leaves it using the same mode
cnt = 0
for t in T:
    for i in I:
        m.addConstr(
            quicksum(x[u, i, t] for u in IN[i]) ==
            quicksum(x[i, v, t] for v in OUT[i])
        )
        cnt += 1
report("flow conservation", cnt)

# each route must be assigned to exactly one depot
cnt = 0
for i in I:
    m.addConstr(
        quicksum(z[i, j] for j in J_cand[i]) == 1
    )
    cnt += 1
report("route–depot assignment", cnt)

# a route can only be assigned to a depot if that depot is opened
cnt = 0
for i, j in z:
    m.addConstr(
        z[i, j] <= y[j]
    )
    cnt += 1
report("z <= y (open depot)", cnt)

# total assigned demand at a depot cannot exceed its capacity
cnt = 0
for j in J:
    m.addConstr(
        0.56 * quicksum(D[i] * z[i, j] for i in I if (i, j) in z)
        <= C_d * y[j]
    )
    cnt += 1
report("depot capacity", cnt)

# total working time per mode must cover depot-to-route travel and service time
# service time includes all handling and refilling required for that depot-route-mode
cnt = 0
for t in T:
    m.addConstr(
        g[t] >= quicksum(
            r[i, t] * z[i, j] * (L.get((j, i, t), 0.0) + S.get((i, j, t), 0.0) + L.get((i, j, t), 0.0))
            for i in I for j in J_cand[i]
        )
    )
    cnt += 1
report("global route duration (depot + service)", cnt)

# total working time per mode must also cover route-to-route travel time
# this accounts for chaining routes sequentially using aggregated flow variables
cnt = 0
for t in T:
    m.addConstr(
        g[t] >= quicksum(
            x[u, v, t] * L[(u, v, t)]
            for (u, v) in A_rr
        )
    )
    cnt += 1
report("global route duration (route–route)", cnt)

# total working time per mode must fit within available vehicles and their maximum uptime
cnt = 0
for t in T:
    m.addConstr(
        g[t] <= n[t] * R_t[t]
    )
    cnt += 1
report("uptime (aggregated)", cnt)

print(f"[ok] total constraints added: {constr_count:,}")

# ============================================================
# 9) solve and the diagnostics
# ============================================================

print("[step 9] optimizing")

m.Params.OutputFlag = 1
m.Params.MIPFocus = 1
m.Params.MIPGap = 0.005
m.Params.DisplayInterval = 1

m.optimize()

sol_path = os.path.join(SCRIPT_DIR, "Solution_small.sol")
m.write(sol_path)

print(f"[ok] solution written to: {sol_path}")


print("[status]", m.Status)
print(f"[info] best gap achieved = {m.MIPGap:.4%}")

if m.Status == GRB.INFEASIBLE:
    print("[diag] model infeasible: computing iis")
    m.computeIIS()
    m.write("model_iis.ilp")

# ============================================================
# 10) EXTENSIVE EXPORT RESULTS 
# ============================================================

print("[step 10] exporting extensive results")

if m.SolCount > 0:
    print(f"[ok] best objective value = {m.ObjVal:.4f}")
    print(f"[ok] optimality gap       = {m.MIPGap:.4%}")
    print(f"[ok] solver status        = {m.Status}")

    # --------------------------------------------------------
    # 1) MODEL-LEVEL SUMMARY
    # --------------------------------------------------------
    model_summary = [{
        "objective_value": m.ObjVal,
        "optimality_gap": m.MIPGap,
        "status": m.Status,
        "num_routes": len(I),
        "num_depots": len(J),
        "num_modes": len(T),
        "num_arcs": len(A),
        "num_variables": m.NumVars,
        "num_constraints": m.NumConstrs,
    }]

    # --------------------------------------------------------
    # 2) DEPOT-LEVEL RESULTS + COSTS
    # --------------------------------------------------------
    depot_rows = []
    depot_cost_rows = []

    for j in J:
        assigned_routes = [i for i in I if (i, j) in z and z[i, j].X > 0.5]

        cost_open = f_j[j] * y[j].X
        cost_handling = sum(0.56 * c_s * D[i] for i in assigned_routes)

        time_est = 0.0
        cost_time_est = 0.0
        for i in assigned_routes:
            mode = next(t for t in T if r[i, t].X > 0.5)
            t_srv = S.get((i, j, mode), 0.0)
            t_out = L.get((j, i, mode), 0.0)
            t_back = L.get((i, j, mode), 0.0)
            t_i = t_out + t_srv + t_back
            time_est += t_i
            cost_time_est += (w + c_t[mode]) * t_i

        depot_rows.append({
            "depot": j,
            "open": int(round(y[j].X)),
            "num_routes_assigned": len(assigned_routes),
            "total_demand": sum(D[i] for i in assigned_routes),
            "capacity_used": sum(D[i] for i in assigned_routes) / C_d if y[j].X > 0 else 0.0
        })

        depot_cost_rows.append({
            "depot": j,
            "cost_open": cost_open,
            "cost_handling": cost_handling,
            "time_estimate": time_est,
            "cost_time_estimate": cost_time_est,
            "cost_total_estimate": cost_open + cost_handling + cost_time_est
        })

    # --------------------------------------------------------
    # 3) ROUTE-LEVEL RESULTS + COSTS
    # --------------------------------------------------------
    route_rows = []
    route_cost_rows = []

    for i in I:
        depot = next(j for j in J_cand[i] if z[i, j].X > 0.5)
        mode = next(t for t in T if r[i, t].X > 0.5)

        t_srv = S.get((i, depot, mode), 0.0)
        t_out = L.get((depot, i, mode), 0.0)
        t_back = L.get((i, depot, mode), 0.0)

        time_est = t_out + t_srv + t_back
        cost_handling = 0.56 * c_s * D[i]
        cost_time_est = (w + c_t[mode]) * time_est

        route_rows.append({
            "route": i,
            "assigned_depot": depot,
            "assigned_mode": mode,
            "demand": D[i],
            "service_time": t_srv,
            "travel_time_from_depot": t_out,
            "travel_time_to_depot": t_back,
            "round_trip_travel_time": t_out + t_back,
            "total_time_estimate": time_est
        })

        route_cost_rows.append({
            "route": i,
            "assigned_depot": depot,
            "assigned_mode": mode,
            "cost_handling": cost_handling,
            "cost_time_estimate": cost_time_est,
            "cost_total_estimate": cost_handling + cost_time_est
        })

    # --------------------------------------------------------
    # 4) MODE-LEVEL (FLEET, TIME, SLACK)
    # --------------------------------------------------------
    mode_rows = []
    uptime_rows = []

    for t in T:
        mode_rows.append({
            "mode": t,
            "vehicles_used": int(round(n[t].X)),
            "total_work_time": g[t].X,
            "max_available_time": n[t].X * R_t[t],
            "utilization": g[t].X / (n[t].X * R_t[t]) if n[t].X > 0 else 0.0,
        })

        uptime_rows.append({
            "mode": t,
            "g": g[t].X,
            "n": n[t].X,
            "R": R_t[t],
            "nR": n[t].X * R_t[t],
            "slack": (n[t].X * R_t[t]) - g[t].X
        })

    # --------------------------------------------------------
    # 5) ROUTE CHAINS (PREDECESSOR / SUCCESSOR)
    # --------------------------------------------------------
    chain_rows = []
    chain_edge_rows = []

    for i in I:
        mode = next(t for t in T if r[i, t].X > 0.5)
        predecessors = [u for u in I if (u, i) in A_rr and x[u, i, mode].X > 0.5]
        successors = [v for v in I if (i, v) in A_rr and x[i, v, mode].X > 0.5]

        chain_rows.append({
            "route": i,
            "mode": mode,
            "is_start": int(len(predecessors) == 0),
            "is_end": int(len(successors) == 0),
            "is_middle": int(len(predecessors) > 0 and len(successors) > 0),
            "predecessors": ",".join(predecessors),
            "successors": ",".join(successors),
        })

    for (u, v) in A_rr:
        for t in T:
            if x[u, v, t].X > 0.5:
                chain_edge_rows.append({
                    "mode": t,
                    "from_route": u,
                    "to_route": v,
                    "flow_count": int(round(x[u, v, t].X)),
                    "travel_time": L.get((u, v, t), None),
                    "distance": d.get((u, v), None)
                })

    # --------------------------------------------------------
    # 6) RECONSTRUCT TOURS PER MODE
    # --------------------------------------------------------
    tour_rows = []

    for t in T:
        start_routes = [
            i for i in I
            if r[i, t].X > 0.5 and
            not any((u, i) in A_rr and x[u, i, t].X > 0.5 for u in I)
        ]

        for start in start_routes:
            tour = [start]
            current = start
            while True:
                nxt = [v for v in I if (current, v) in A_rr and x[current, v, t].X > 0.5]
                if not nxt:
                    break
                current = nxt[0]
                tour.append(current)

            tour_rows.append({
                "mode": t,
                "num_routes": len(tour),
                "tour_sequence": " -> ".join(tour)
            })

    # --------------------------------------------------------
    # 7) COST BREAKDOWN (MODEL LEVEL)
    # --------------------------------------------------------
    cost_rows = [{
        "cost_depots": sum(f_j[j] * y[j].X for j in J),
        "cost_handling": 0.56 * sum(c_s * D[i] for i in I),
        "cost_startup": 0.5 * sum(
            c_t_s[t] * x[j, i, t].X
            for (j, i) in A_dr for t in T
        ),
        "cost_labor": sum((c_t[t] + w) * g[t].X for t in T),
        "total_cost": m.ObjVal
    }]

    # --------------------------------------------------------
    # WRITE TO EXCEL
    # --------------------------------------------------------
    out_path = os.path.join(SCRIPT_DIR, "solution_US.xlsx")

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        pd.DataFrame(model_summary).to_excel(writer, sheet_name="ModelSummary", index=False)
        pd.DataFrame(depot_rows).to_excel(writer, sheet_name="Depots", index=False)
        pd.DataFrame(depot_cost_rows).to_excel(writer, sheet_name="DepotCosts", index=False)
        pd.DataFrame(route_rows).to_excel(writer, sheet_name="Routes", index=False)
        pd.DataFrame(route_cost_rows).to_excel(writer, sheet_name="RouteCosts", index=False)
        pd.DataFrame(mode_rows).to_excel(writer, sheet_name="Modes", index=False)
        pd.DataFrame(uptime_rows).to_excel(writer, sheet_name="UptimeSlack", index=False)
        pd.DataFrame(chain_rows).to_excel(writer, sheet_name="RouteChains", index=False)
        pd.DataFrame(chain_edge_rows).to_excel(writer, sheet_name="ChainEdges", index=False)
        pd.DataFrame(tour_rows).to_excel(writer, sheet_name="Tours", index=False)
        pd.DataFrame(cost_rows).to_excel(writer, sheet_name="CostBreakdown", index=False)

    print(f"[ok] extensive export written to: {out_path}")

else:
    print("[warn] no feasible solution — nothing exported")
    


