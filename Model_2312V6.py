# ============================================================
# MILP DEPOT + ROUTING MODEL
# Depot-indexed vehicles + sparse arcs (KNN)
# PC-RUNNABLE VERSION WITH DIAGNOSTICS
#
# IMPLEMENTS:
#  1) Depot-indexed vehicles with per-type maxima (NO global K)
#  2) Aggressive arc sparsification (KNN)
#  3) Sparse time variables (t_arr) aligned with routing
# ============================================================

import os
import sys
import pickle
import pandas as pd
from gurobipy import Model, GRB, quicksum

# ============================================================
# 0) ENVIRONMENT CHECK
# ============================================================

print("=" * 60)
print("STARTING MODEL EXECUTION")
print("Python version:", sys.version)
print("Working directory:", os.getcwd())
print("=" * 60, "\n")


# ============================================================
# 1) LOAD DATA
# ============================================================

print("[STEP 1] Loading model_input.pkl ...")

with open("model_input.pkl", "rb") as f:
    model_data = pickle.load(f)

I = list(model_data["sets"]["I"])
J = list(model_data["sets"]["J"])
T = list(model_data["sets"]["T"])

d = model_data["parameters"]["d"]
D = model_data["parameters"]["D"]
S = model_data["parameters"]["service_time"]

print(f"[INFO] |I|={len(I)}, |J|={len(J)}, |T|={len(T)}")
print("[OK] Data loaded\n")

# ============================================================
# 2) PARAMETERS
# ============================================================

print("[STEP 2] Building parameters...")

K_MAX_T = {
    "Car": 20,
    "Moped": 20,
    "EBikeCart": 20,
    "FootBike": 20
}

K_MAX = sum(K_MAX_T[t] for t in T)
K = range(K_MAX)

print(f"[INFO] Vehicle slots per depot: {K_MAX}")

f_j = {j: 25.0 for j in J}
c_s = 1.0
c_t = {"Car": 40, "Moped": 30, "EBikeCart": 25, "FootBike": 20}
w = 35.0

Q_t = {"Car": 20, "Moped": 6, "EBikeCart": 6, "FootBike": 3}
C_d = 2000

v_t = {"Car": 40, "Moped": 30, "EBikeCart": 20, "FootBike": 15}
R_t = {"Car": 8.0, "Moped": 7.5, "EBikeCart": 7.0, "FootBike": 6.0}
R_max = max(R_t.values())

L = {(i, j, t): d[i, j] / v_t[t] for (i, j) in d for t in T}

M = 1e5

print("[OK] Parameters built\n")

# ============================================================
# 3) BUILD SPARSE ARC SET (KNN)
# ============================================================

print("[STEP 3] Building sparse arc set (k-nearest neighbors)...")

KNN = 15

A_rr = set()  # route -> route
A_dr = set()  # depot -> route

for i in I:
    neigh = sorted(
        [h for h in I if (i, h) in d and i != h],
        key=lambda h: d[i, h]
    )[:KNN]
    for h in neigh:
        A_rr.add((i, h))

for j in J:
    neigh = sorted(
        [i for i in I if (j, i) in d],
        key=lambda i: d[j, i]
    )[:KNN]
    for i in neigh:
        A_dr.add((j, i))

print(f"[INFO] |A_rr| (route→route) = {len(A_rr)}")
print(f"[INFO] |A_dr| (depot→route) = {len(A_dr)}")
print("[OK] Sparse arc sets built\n")

# ============================================================
# 3B) GUARANTEE ARC COVERAGE (MINIMAL FEASIBILITY PATCH)
# ============================================================
# Ensures every route i has at least:
#  - one depot->route option OR one route->route incoming option (for Assignment)
#  - one route->route outgoing option (helps Flow/Time)
#
# This keeps the KNN idea but prevents "0 == 1" infeasibilities.

print("[STEP 3B] Ensuring every route has feasible sparse connectivity...")

added_dr = 0
added_rr_in = 0
added_rr_out = 0

A_dr_by_route = {i: [] for i in I}
A_rr_in_by_route = {i: [] for i in I}
A_rr_out_by_route = {i: [] for i in I}

for (j, i) in A_dr:
    A_dr_by_route[i].append(j)
for (h, i) in A_rr:
    A_rr_in_by_route[i].append(h)
    A_rr_out_by_route[h].append(i)

for i in I:
    has_incoming = (len(A_dr_by_route[i]) > 0) or (len(A_rr_in_by_route[i]) > 0)
    if not has_incoming:
        cand = [(j, d[j, i]) for j in J if (j, i) in d]
        if cand:
            j_star = min(cand, key=lambda x: x[1])[0]
            A_dr.add((j_star, i))
            A_dr_by_route[i].append(j_star)
            added_dr += 1
        else:
            cand2 = [(h, d[h, i]) for h in I if h != i and (h, i) in d]
            if cand2:
                h_star = min(cand2, key=lambda x: x[1])[0]
                A_rr.add((h_star, i))
                A_rr_in_by_route[i].append(h_star)
                A_rr_out_by_route[h_star].append(i)
                added_rr_in += 1

for i in I:
    if len(A_rr_out_by_route[i]) == 0:
        cand = [(h, d[i, h]) for h in I if h != i and (i, h) in d]
        if cand:
            h_star = min(cand, key=lambda x: x[1])[0]
            A_rr.add((i, h_star))
            A_rr_out_by_route[i].append(h_star)
            A_rr_in_by_route[h_star].append(i)
            added_rr_out += 1

print(f"[INFO] Added depot->route arcs: {added_dr}")
print(f"[INFO] Added route->route incoming arcs: {added_rr_in}")
print(f"[INFO] Added route->route outgoing arcs: {added_rr_out}")
print(f"[INFO] New |A_rr|={len(A_rr)}, New |A_dr|={len(A_dr)}")
print("[OK] Connectivity patch applied\n")

# ============================================================
# 3B-DIAG) PRESOLVE GRAPH FEASIBILITY DIAGNOSTICS
# ============================================================

print("[DIAG] Checking sparse-graph feasibility BEFORE model build...")

# Routes with no depot->route arc at all
no_depot_start = [
    i for i in I
    if not any((j, i) in A_dr for j in J)
]

print(f"[DIAG] Routes with NO depot->route arc: {len(no_depot_start)}")
if no_depot_start:
    print("   sample:", no_depot_start[:20])

# Routes that never appear in any depot's START_ARCS
START_ARCS_tmp = {}
for (j, i) in A_dr:
    START_ARCS_tmp.setdefault(j, []).append(i)

not_in_any_start_arcs = [
    i for i in I
    if not any(i in START_ARCS_tmp.get(j, []) for j in J)
]

print(f"[DIAG] Routes not in ANY START_ARCS[j]: {len(not_in_any_start_arcs)}")
if not_in_any_start_arcs:
    print("   sample:", not_in_any_start_arcs[:20])

print("[DIAG] End sparse-graph diagnostic\n")


# ============================================================
# 3C) IDENTIFY + REMOVE ROUTES WITH NO DEPOT ACCESS
# ============================================================

print("[STEP 3C] Identifying routes unreachable from any depot...")

unreachable_routes = set()

for i in I:
    has_depot_access = any((j, i) in A_dr for j in J)
    if not has_depot_access:
        unreachable_routes.add(i)

print(f"[CHECK] Found {len(unreachable_routes)} routes with NO depot access.")
for i in list(unreachable_routes)[:20]:
    print("   ", i)
if len(unreachable_routes) > 20:
    print("   ...")


# --- HARD SKIP: remove from the modeling universe ---
if unreachable_routes:
    I = [i for i in I if i not in unreachable_routes]

    # Remove arcs touching unreachable routes
    A_dr = {(j, i) for (j, i) in A_dr if i not in unreachable_routes}
    A_rr = {(i, h) for (i, h) in A_rr if i not in unreachable_routes and h not in unreachable_routes}

    # Remove demands too (prevents export loops / z-index errors)
    for i in unreachable_routes:
        if i in D:
            del D[i]

print(f"[OK] After removal: |I|={len(I)}, |A_rr|={len(A_rr)}, |A_dr|={len(A_dr)}\n")

# Keep record for reporting (outside the model)
removed_routes = unreachable_routes.copy()

print(f"[SUMMARY] Removed {len(removed_routes)} routes from optimization.")


# ============================================================
# 4) BUILD SPARSE TIME INDEX SET (AFTER CLEANING)
# ============================================================

print("[STEP 4] Building sparse time index set...")

T_IDX = set()

for (j, i) in A_dr:
    for k in K:
        T_IDX.add((i, j, k))

for (h, i) in A_rr:
    for (j2, h2) in A_dr:
        if h2 == h:
            for k in K:
                T_IDX.add((i, j2, k))

print(f"[INFO] |T_IDX| (time vars) = {len(T_IDX)}")
print("[OK] Sparse time index built\n")

# ============================================================
# 5) MODEL
# ============================================================

print("[STEP 5] Creating Gurobi model...")
m = Model("DepotRouting_DepotIndexed_Sparse_FullFixed")
print("[OK] Model created\n")

# ============================================================
# 6) VARIABLES
# ============================================================

print("[STEP 6] Adding decision variables...")

y = m.addVars(J, vtype=GRB.BINARY, name="y")

u = m.addVars(J, K, vtype=GRB.BINARY, name="u")
a = m.addVars(J, K, T, vtype=GRB.BINARY, name="a")

x_start = m.addVars(A_dr, K, vtype=GRB.BINARY, name="x_start")
x_route = m.addVars(A_rr, K, vtype=GRB.BINARY, name="x_route")

z = m.addVars(I, J, vtype=GRB.BINARY, name="z")

t_arr = m.addVars(T_IDX, lb=0, vtype=GRB.CONTINUOUS, name="t")
g = m.addVars(J, K, lb=0, vtype=GRB.CONTINUOUS, name="g")

print("[OK] Variables added\n")

# ============================================================
# 7) OBJECTIVE FUNCTION
# ============================================================

print("[STEP 7] Setting objective function...")

m.setObjective(
    quicksum(f_j[j] * y[j] for j in J)
    + 0.56 * quicksum(c_s * D[i] * z[i, j] for i in I for j in J)
    + quicksum(
        g[j, k] * (quicksum(c_t[t] * a[j, k, t] for t in T) + w)
        for j in J for k in K
    ),
    GRB.MINIMIZE
)

print("[OK] Objective set\n")

# ============================================================
# 8) CONSTRAINTS
# ============================================================

from collections import defaultdict

OUT_ARCS = defaultdict(list)
IN_ARCS  = defaultdict(list)
START_ARCS = defaultdict(list)

for (i, h) in A_rr:
    OUT_ARCS[i].append(h)
    IN_ARCS[h].append(i)

for (j, i) in A_dr:
    START_ARCS[j].append(i)

print("[STEP 8] Adding constraints...")

# Vehicle logic
for j in J:
    for k in K:
        m.addConstr(quicksum(a[j, k, t] for t in T) == u[j, k])
        m.addConstr(g[j, k] <= R_max * u[j, k])
        m.addConstr(u[j, k] <= y[j])

for j in J:
    for t in T:
        m.addConstr(quicksum(a[j, k, t] for k in K) <= K_MAX_T[t] * y[j])

print("  [OK] Vehicle logic")

# (NEW) Each vehicle slot k may belong to at most one depot
for k in K:
    m.addConstr(
        quicksum(u[j, k] for j in J) <= 1,
        name=f"one_depot_per_vehicle_{k}"
    )

print("  [OK] NEW Constraint: one depot per vehicle slot")

# (1) Assignment (ROBUST: only for remaining reachable I)
for i in I:
    m.addConstr(
        quicksum(x_start[j, i, k] for (j, i2) in A_dr if i2 == i for k in K)
        + quicksum(x_route[h, i, k] for (h, i2) in A_rr if i2 == i for k in K)
        <= 1,
        name=f"assign_{i}"
    )

print("  [OK] Constraint (1) Assignment")

# (NEW) Route–depot linking (sparse version of constraint (2))
for j in J:
    start_routes = START_ARCS.get(j, [])
    if not start_routes:
        continue
    for i in start_routes:
        m.addConstr(
            quicksum(x_start[j, i, k] for k in K)
            + quicksum(x_route[h, i, k] for h in IN_ARCS.get(i, []) for k in K)
            <= len(K) * z[i, j],
            name=f"route_depot_link_{i}_{j}"
        )

print("  [OK] NEW Constraint: route–depot linking")

# (3) Flow conservation (kept as you had)
print("  [BUILD] Constraint (3) Flow conservation (FAST, SAFE)")

for j in J:
    start_routes = START_ARCS.get(j, [])
    if not start_routes:
        continue
    for k in K:
        for i in start_routes:
            m.addConstr(
                quicksum(x_route[i, h, k] for h in OUT_ARCS.get(i, []))
                ==
                quicksum(x_route[h, i, k] for h in IN_ARCS.get(i, []))
                + x_start[j, i, k],
                name=f"flow_{j}_{k}_{i}"
            ) 

print("  [OK] Constraint (3) Flow conservation")

# (NEW) A vehicle may only use route→route arcs if it has started
for k in K:
    m.addConstr(
        quicksum(x_route[i, h, k] for (i, h) in A_rr)
        <=
        quicksum(x_start[j, i, k] for (j, i) in A_dr),
        name=f"no_route_without_start_{k}"
    )

print("  [OK] NEW Constraint: no route without start")

# (4) Single start
for j in J:
    for k in K:
        m.addConstr(
            quicksum(x_start[j, i, k] for (j2, i) in A_dr if j2 == j) <= 1
        )

print("  [OK] Constraint (4) Single start")

# (5) Vehicle capacity
print("  [BUILD] Constraint (5) Vehicle capacity (FAST, SAFE)")

for j in J:
    start_routes = START_ARCS.get(j, [])
    if not start_routes:
        continue
    for k in K:
        m.addConstr(
            quicksum(
                D[i] * (
                    x_start[j, i, k]
                    + quicksum(x_route[h, i, k] for h in IN_ARCS.get(i, []))
                )
                for i in start_routes
            )
            <= quicksum(Q_t[t] * a[j, k, t] for t in T),
            name=f"capacity_{j}_{k}"
        )

print("  [OK] Constraint (5) Capacity")

# (6) Depot capacity
for j in J:
    m.addConstr(0.56 * quicksum(D[i] * z[i, j] for i in I) <= C_d * y[j])

print("  [OK] Constraint (6) Depot capacity")

# (8) Time continuity (route -> route)  <<< COMMENTED OUT (DO NOT DELETE)
print("  [BUILD] Constraint (8) Time continuity (COMMENTED OUT)")

# for j in J:
#     start_routes = START_ARCS.get(j, [])
#     if not start_routes:
#         continue
#     for k in K:
#         for i in start_routes:
#             for h in OUT_ARCS.get(i, []):
#                 m.addConstr(
#                     t_arr[i, j, k]
#                     + quicksum(a[j, k, t] * S.get((i, h, t), 0) for t in T)
#                     + quicksum(a[j, k, t] * L.get((i, h, t), 0) for t in T)
#                     - M * (1 - x_route[i, h, k])
#                     <= t_arr[h, j, k],
#                     name=f"time_{j}_{k}_{i}_{h}"
#                 )

print("  [OK] Constraint (8) Time continuity (SKIPPED)")

# (10) Vehicle uptime
for j in J:
    for k in K:
        m.addConstr(g[j, k] <= quicksum(a[j, k, t] * R_t[t] for t in T))

print("  [OK] Constraint (10) Vehicle uptime\n")

# ============================================================
# 9) SOLVE
# ============================================================

print("[STEP 9] Starting optimization...")

# ================= PRESOLVE / MEMORY CONTROL =================
m.Params.Presolve = 1          # conservative presolve
m.Params.Threads = 1           # limit memory usage
m.Params.Method = 1            # dual simplex (lower RAM)
m.Params.NodefileStart = 0.5   # start writing nodes to disk early
m.Params.Aggregate = 1
m.Params.DualReductions = 0
# =============================================================

m.optimize()

m.optimize()

print("[OK] Optimization finished with status:", m.status, "\n")

# ============================================================
# DIAGNOSTIC BLOCK: DISAMBIGUATE INF vs UNBD + FEASRELAX
# ============================================================

if m.status == GRB.INF_OR_UNBD:
    print("\n[DIAG] Status is INF_OR_UNBD (4). Disambiguating...")
    m.Params.DualReductions = 0
    m.Params.InfUnbdInfo = 1
    print("[DIAG] Re-optimizing with DualReductions=0, InfUnbdInfo=1 ...")
    m.optimize()
    print("[DIAG] New status:", m.status)

if m.status == GRB.INFEASIBLE:
    print("\n[DIAG] Model is INFEASIBLE.")
    print("[DIAG] Full IIS is too big and can run out of memory on this model.")
    print("[DIAG] Running FeasRelax to identify what must be relaxed...")

    mr = m.copy()
    mr.Params.OutputFlag = 1
    mr.Params.DisplayInterval = 1

    mr.feasRelaxS(relaxobjtype=0, minrelax=True, vrelax=False, crelax=True)
    mr.optimize()

    print("[DIAG] FeasRelax status:", mr.status)
    if mr.status == GRB.OPTIMAL:
        print("[DIAG] FeasRelax objective (total violation):", mr.ObjVal)

        viols = []
        for c in mr.getConstrs():
            s = c.Slack
            if abs(s) > 1e-6:
                viols.append((abs(s), c.ConstrName, s))

        viols.sort(reverse=True)
        print("[DIAG] Top violated constraints (abs(slack), name, slack):")
        for v in viols[:30]:
            print("   ", v)
    else:
        print("[DIAG] FeasRelax did not solve to optimality; try reducing KNN / model size.")

elif m.status == GRB.UNBOUNDED:
    print("\n[DIAG] Model is UNBOUNDED. (Unexpected here: costs are positive.)")
    print("[DIAG] Check missing upper bounds / missing linking of variables to objective.")

# ============================================================
# 10) EXPORT RESULTS
# ============================================================

print("[STEP 10] Exporting results...")

if m.status == GRB.OPTIMAL:

    depot_rows = []
    vehicle_rows = []

    for j in J:
        depot_rows.append({
            "depot": j,
            "open": y[j].X,
            "assigned_demand": sum(D[i] * z[i, j].X for i in I)
        })

    for j in J:
        for k in K:
            if u[j, k].X > 0.5:
                ttype = next(t for t in T if a[j, k, t].X > 0.5)
                vehicle_rows.append({
                    "vehicle_id": f"{j}_{k}",
                    "depot": j,
                    "vehicle_type": ttype,
                    "route_duration": g[j, k].X
                })

    with pd.ExcelWriter("solution_details_sparse_fixed.xlsx", engine="xlsxwriter") as writer:
        pd.DataFrame(vehicle_rows).to_excel(writer, "VehicleSummary", index=False)
        pd.DataFrame(depot_rows).to_excel(writer, "Depots", index=False)

    print("✅ Solution exported to solution_details_sparse_fixed.xlsx")

else:
    print("❌ Model not optimal – no export")

print("\n" + "=" * 60)
print("SCRIPT FINISHED")
print("=" * 60)
