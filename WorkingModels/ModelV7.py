'''
depot + routing milp
aggregated vehicles with route–mode assignment

this model:
 - uses sparse routing arcs to reduce size
 - allows depot-dependent service times
 - sizes the fleet per transport mode (aggregated)
 - assigns each route to exactly one depot and one mode
 - enforces depot-consistent routes 
 
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
# load preprocessed model input from pickle
# this contains all sets and parameters used in the model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_NAME = "model_input_US.pkl"

print(f"[step 1] loading data from {PKL_NAME}")

with open(os.path.join(SCRIPT_DIR, PKL_NAME), "rb") as f:
    model_data = pickle.load(f)

# sets
I = list(model_data["sets"]["I"])    # routes / delivery clusters
J = list(model_data["sets"]["J"])    # depots
T = list(model_data["sets"]["T"])    # transport modes

# parameters
D = model_data["parameters"]["D"]                # demand per route
d = model_data["parameters"]["d"]                # distance matrix
S = model_data["parameters"]["service_time"]     # service time per (route, depot, mode)

print(f"[ok] |I|={len(I)}, |J|={len(J)}, |T|={len(T)}")
print(f"[ok] |distance arcs|={len(d)}, |service entries|={len(S)}")

# ============================================================
# 2) parameters
# ============================================================
# all cost, capacity, and operational parameters

print("[step 2] building parameters")

c_s = 0.75                         # storage / handling cost per delivery bag
f_j = {j: 50 for j in J}          # fixed depot opening costs 
C_d = 10000                        # depot capacity (non-binding)

# fixed daily vehicle costs
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

'''
vehicle capacities --> not used in current version due to this being 
incorporated in the regression for service times with the depot reloads.

i.e.: every mode is able to serve every route but with different service 
times due to some modes needing to return to the depot in the middle of the route to reload.

Q_t = {
    "FootBike": 3,
    "Car": 25,
    "EBikeCart": 5,
    "Moped": 3
}

'''
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

# M = 10000                          # big-m constant (not used directly)
w = 23.30                          # labor cost per hour

# ============================================================
# 3) candidate depots
# ============================================================

print("[step 3] selecting candidate depots")

# maximum number of candidate depots per route
# this reduces model size while keeping nearby depots available
N_DEPOTS = 20

# mapping: route i -> list of candidate depots
J_cand = {}

# counter for routes without any depot distances --> some datapoints are not consistent with reality so they are left out
missing = 0

for i in I:
    # collect all depots j with a known distance to route i
    cand = [(j, d[(i, j)]) for j in J if (i, j) in d]

    # sort candidate depots by increasing distance
    cand.sort(key=lambda x: x[1])

    # if no depot has a distance to this route, mark as missing
    if not cand:
        missing += 1

    # keep only the N closest depots
    J_cand[i] = [j for (j, _) in cand[:N_DEPOTS]]

print(f"[ok] candidate depots per route = {N_DEPOTS}")
print(f"[check] routes without depot distances: {missing}")

# identify routes that have no candidate depots at all
bad_routes = [i for i in I if len(J_cand[i]) == 0]

# remove unreachable routes from all relevant structures
if bad_routes:
    print(f"[warn] removing {len(bad_routes)} unreachable routes:")
    for i in bad_routes:
        print("   ", i)

    # remove routes from the active route set
    I = [i for i in I if i not in bad_routes]

    # remove demand entries of removed routes
    D = {i: D[i] for i in I}

    # keep candidate depots only for remaining routes
    J_cand = {i: J_cand[i] for i in I}

print(f"[ok] remaining routes after cleanup: {len(I)}")


# ============================================================
# 4) sparse arcs
# ============================================================
# build a reduced arc set:
# - depot → route
# - route → route (nearest neighbors)
# - route → depot

print("[step 4] building sparse arc set")

A_dr, A_rd, A_rr = [], [], []

# depot ↔ route arcs
for i in I:
    for j in J_cand[i]:
        if (j, i) in d:
            A_dr.append((j, i))
        if (i, j) in d:
            A_rd.append((i, j))

# route → route arcs (k-nearest neighbors)
NEIGH_ROUTES = 15
rr_by_u = defaultdict(list)

for (u, v), dist in d.items():
    if u in I and v in I and u != v:
        rr_by_u[u].append((v, dist))

for u in I:
    rr_by_u[u].sort(key=lambda x: x[1])
    for v, _ in rr_by_u[u][:NEIGH_ROUTES]:
        A_rr.append((u, v))

# combined arc set
A = list(dict.fromkeys(A_dr + A_rr + A_rd))

# adjacency lists
OUT = defaultdict(list)
IN = defaultdict(list)
for (u, v) in A:
    OUT[u].append(v)
    IN[v].append(u)

# ============================================================
# 5) travel times
# ============================================================
# convert distances to travel times per mode

print("[step 5] precomputing travel times")

L = {(u, v, t): d[(u, v)] / v_t[t] for (u, v) in A for t in T}

# ============================================================
# 6) model + variables
# ============================================================

print("[step 6] creating model and variables")

m = Model("DepotRoutingAggregated")

# x[u,v,t]: number of vehicles of mode t traveling arc (u,v)
x = m.addVars(A, T, vtype=GRB.BINARY, lb=0, name="x") #-----> dit moet worden veranderd naar binary er kunnen niet meer vehicles over 1 arc gaan

# r[i,t]: route i is served by mode t
r = m.addVars(I, T, vtype=GRB.BINARY, name="r")

# y[j]: depot j is opened
y = m.addVars(J, vtype=GRB.BINARY, name="y")

# z[i,j]: route i assigned to depot j
z = m.addVars([(i, j) for i in I for j in J_cand[i]], vtype=GRB.BINARY, name="z")

# g[t]: total working time of mode t
g = m.addVars(T, lb=0, name="g")

# n[t]: number of vehicles of mode t used
n = m.addVars(T, vtype=GRB.INTEGER, lb=0, name="n")

n_jt = m.addVars(J, T, vtype=GRB.INTEGER, lb=0, name="n_jt")

# ============================================================
# 7) objective
# ============================================================
# minimize total system cost:
# - depot opening
# - handling costs
# - vehicle startup costs
# - labor and operational time costs


print("[step 7] setting objective")

m.setObjective(
    quicksum(f_j[j] * y[j] for j in J)
    + 0.56 * quicksum(c_s * D[i] * z[i, j] for (i, j) in z)
    + 0.5 * quicksum(c_t_s[t] * n_jt[j, t] for j in J for t in T)
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
    print(f"  [ok] {name:<45s}: {n:>8,d}   (total: {constr_count:>8,d})")

# 8.1 each route must be visited exactly once
for i in I:
    m.addConstr(quicksum(x[u, i, t] for u in IN[i] for t in T) == 1)
report("8.1 assignment (visit once)", len(I))

# 8.2 each route chooses exactly one transport mode
for i in I:
    m.addConstr(quicksum(r[i, t] for t in T) == 1)
report("8.2 route–mode assignment", len(I))

# 8.3 flow into a route using mode t only if that mode is selected
for i in I:
    for t in T:
        m.addConstr(quicksum(x[u, i, t] for u in IN[i]) <= r[i, t])
report("8.3 flow–mode linking", len(I) * len(T))

# 8.4 flow conservation per route and per mode
# vehicles entering a route must also leave it using the same mode
for t in T:
    for i in I:
        m.addConstr(
            quicksum(x[u, i, t] for u in IN[i]) ==
            quicksum(x[i, v, t] for v in OUT[i])
        )
report("8.4 flow conservation", len(I) * len(T))

# 8.5 each route is assigned to exactly one depot
for i in I:
    m.addConstr(quicksum(z[i, j] for j in J_cand[i]) == 1)
report("8.5 route–depot assignment", len(I))

# 8.6 routes can only be assigned to opened depots
for (i, j) in z:
    m.addConstr(z[i, j] <= y[j])
report("8.6 z <= y", len(z))

# 8.7 depot capacity constraint
for j in J:
    m.addConstr(
        0.56 * quicksum(D[i] * z[i, j] for i in I if (i, j) in z)
        <= C_d * y[j]
    )
report("8.7 depot capacity", len(J))

# 8.8 depot arc consistency
# vehicles may only start or end at the depot assigned to the route
for (j, i) in A_dr:
    for t in T:
        m.addConstr(x[j, i, t] <= z[i, j])
for (i, j) in A_rd:
    for t in T:
        m.addConstr(x[i, j, t] <= z[i, j])
report("8.8 depot arc consistency", len(A_dr) * len(T) + len(A_rd) * len(T))

# vehicles per depot per mode equals number of tours that start at that depot in that mode
for j in J:
    for t in T:
        m.addConstr(
            n_jt[j, t] == quicksum(x[j, i, t] for (j2, i) in A_dr if j2 == j)
        )
report("vehicles per depot/mode (start count)", len(J) * len(T))

# (8.xx) total vehicles per mode
# total number of vehicles of mode t equals sum over all depots
for t in T:
    m.addConstr(
        n[t] == quicksum(n_jt[j, t] for j in J)
    )
report("total vehicles per mode", len(T))


# 8.9 depot start = depot end for all modes
# every vehicle that starts at a depot must also end at the same depot
for t in T:
    for j in J:
        m.addConstr(
            quicksum(x[j, i, t] for (j2, i) in A_dr if j2 == j) ==
            quicksum(x[i, j, t] for (i, j2) in A_rd if j2 == j)
        )
report("8.9 depot start/end balance", len(J) * len(T))

# 8.10 global working time accounting per mode
# includes depot-to-route, route-to-route, service, and return-to-depot travel
for t in T:
    m.addConstr(
        g[t] >=
        quicksum(x[j, i, t] * L[(j, i, t)] for (j, i) in A_dr) +
        quicksum(x[u, v, t] * L[(u, v, t)] for (u, v) in A_rr) +
        quicksum(x[i, j, t] * L[(i, j, t)] for (i, j) in A_rd) +
        quicksum(
            r[i, t] * z[i, j] * S.get((i, j, t), 0.0)
            for i in I for j in J_cand[i]
        )
    )
report("8.10 global time accounting", len(T))

# 8.11 uptime constraint
# total working time may not exceed available vehicles times max uptime
for t in T:
    m.addConstr(g[t] <= n[t] * R_t[t])
report("8.11 uptime", len(T))

print(f"[ok] total constraints added: {constr_count:,}")



# ============================================================
# 9) solve
# ============================================================

print("[step 9] optimizing")

m.Params.OutputFlag = 1
m.Params.MIPGap = 0.001
m.Params.Method = 3

m.optimize()

sol_path = os.path.join(SCRIPT_DIR, "Solution_Small.sol")
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
    # 5) FAST ROUTE CHAINS + EDGES (no O(|I|^2))
    # --------------------------------------------------------
    succ = {(i, t): [] for i in I for t in T}
    pred = {(i, t): [] for i in I for t in T}
    chain_edge_rows = []

    for (u, v) in A_rr:
        for t in T:
            val = x[u, v, t].X
            if val > 0.5:
                succ[(u, t)].append(v)
                pred[(v, t)].append(u)
                chain_edge_rows.append({
                    "mode": t,
                    "from_route": u,
                    "to_route": v,
                    "flow_count": int(round(val)),
                    "travel_time": L.get((u, v, t), None),
                    "distance": d.get((u, v), None)
                })

    chain_rows = []
    for i in I:
        mode = next(t for t in T if r[i, t].X > 0.5)
        chain_rows.append({
            "route": i,
            "mode": mode,
            "is_start": int(len(pred[(i, mode)]) == 0),
            "is_end": int(len(succ[(i, mode)]) == 0),
            "is_middle": int(len(pred[(i, mode)]) > 0 and len(succ[(i, mode)]) > 0),
            "predecessors": ",".join(pred[(i, mode)]),
            "successors": ",".join(succ[(i, mode)]),
        })

    # --------------------------------------------------------
    # 6) RECONSTRUCT TOURS PER MODE (FAST)
    # --------------------------------------------------------
    tour_rows = []

    for t in T:
        start_routes = [i for i in I if r[i, t].X > 0.5 and len(pred[(i, t)]) == 0]

        for start in start_routes:
            tour = [start]
            current = start
            visited = set(tour)

            while succ[(current, t)]:
                nxt = succ[(current, t)][0]
                if nxt in visited:
                    break
                tour.append(nxt)
                visited.add(nxt)
                current = nxt

            tour_rows.append({
                "mode": t,
                "num_routes": len(tour),
                "tour_sequence": " -> ".join(tour)
            })

    # --------------------------------------------------------
    # 7) DEPOT START / END BALANCE CHECK
    # --------------------------------------------------------
    A_dr_set = set(A_dr)
    A_rd_set = set(A_rd)

    depot_balance_rows = []
    for t in T:
        for j in J:
            starts = sum(x[j, i, t].X for (j2, i) in A_dr_set if j2 == j)
            ends = sum(x[i, j, t].X for (i, j2) in A_rd_set if j2 == j)

            if starts > 1e-6 or ends > 1e-6:
                depot_balance_rows.append({
                    "depot": j,
                    "mode": t,
                    "starts_from_depot": starts,
                    "ends_at_depot": ends,
                    "difference_starts_minus_ends": starts - ends
                })

    # --------------------------------------------------------
    # 8) COST BREAKDOWN (MODEL LEVEL)
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
    # 9) DETAILED COST BREAKDOWN (REPORTING)
    # --------------------------------------------------------

    # fixed depot opening costs
    cost_fixed_depots = sum(
        f_j[j] * y[j].X
        for j in J
    )

    # variable depot storage / handling costs (based on demand)
    cost_variable_depot_storage = 0.56 * sum(
        c_s * D[i]
        for i in I
    )

    fixed_vehicle_storage_costs = 0.5 * sum(
        c_t_s[t] * n_jt[j, t].X
        for j in J for t in T
    )


    # variable routing costs: vehicle operational cost + mail carrier wage
    cost_variable_routing = sum(
        (c_t[t] + w) * g[t].X
        for t in T
    )

    cost_breakdown_detailed = [{
        "fixed_depot_opening_costs": cost_fixed_depots,
        "variable_depot_storage_costs": cost_variable_depot_storage,
        "fixed_vehicle_storage_costs": fixed_vehicle_storage_costs,
        "variable_routing_costs": cost_variable_routing,
        "total_cost": (
            cost_fixed_depots
            + cost_variable_depot_storage
            + fixed_vehicle_storage_costs
            + cost_variable_routing
        )
    }]


    # --------------------------------------------------------
    # WRITE TO EXCEL
    # --------------------------------------------------------
    out_path = os.path.join(SCRIPT_DIR, "solution_small.xlsx")

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        pd.DataFrame(model_summary).to_excel(writer, sheet_name="ModelSummary", index=False)
        pd.DataFrame(depot_rows).to_excel(writer, sheet_name="Depots", index=False)
        pd.DataFrame(depot_cost_rows).to_excel(writer, sheet_name="DepotCosts", index=False)
        pd.DataFrame(route_rows).to_excel(writer, sheet_name="Routes", index=False)
        pd.DataFrame(route_cost_rows).to_excel(writer, sheet_name="RouteCosts", index=False)
        pd.DataFrame(mode_rows).to_excel(writer, sheet_name="Modes", index=False)
        pd.DataFrame(cost_breakdown_detailed).to_excel(writer, sheet_name="CostBreakdownDetailed", index=False)


    print(f"[ok] extensive export written to: {out_path}")

else:
    print("[warn] no feasible solution — nothing exported")
