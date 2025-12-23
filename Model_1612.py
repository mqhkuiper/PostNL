import pickle
import math
from gurobipy import Model, GRB, quicksum

# ============================================================
# 0) LOAD DATA
# ============================================================

with open("model_input.pkl", "rb") as f:
    model_data = pickle.load(f)

# -----------------------------
# Sets
# -----------------------------
I = model_data["sets"]["I"]          # routes
J = model_data["sets"]["J"]          # depots
V = model_data["sets"]["V"]          # nodes
T = model_data["sets"]["T"]          # vehicle types

# -----------------------------
# Parameters
# -----------------------------
d = model_data["parameters"]["d"]                    # km
D = model_data["parameters"]["D"]                    # bags
service_time = model_data["parameters"]["service_time"]  # (i,j,t) hours

total_demand = sum(D.values())

# ============================================================
# 1) VEHICLE PARAMETERS
# ============================================================

v_t = {
    "Car": 40.0,
    "Moped": 30.0,
    "EBikeCart": 20.0,
    "FootBike": 15.0,
}

Q_t = {
    "Car": 20,
    "Moped": 6,
    "EBikeCart": 6,
    "FootBike": 3,
}

R_t = {
    "Car": 8.0,
    "Moped": 7.5,
    "EBikeCart": 7.0,
    "FootBike": 6.0,
}

# ------------------------------------------------------------
# Aggregated vehicle index K_t (ROUTE INDEX, NOT VEHICLE ID)
# ------------------------------------------------------------
K = {
    t: list(range(
        math.ceil(total_demand / Q_t[t])
    ))
    for t in T
}

# ============================================================
# 2) ARC SET (FILTERED)
# ============================================================
A = []
skipped_arcs = []

for (i, j) in d.keys():
    if i in V and j in V:
        A.append((i, j))
    else:
        skipped_arcs.append((i, j))

print("\n=== ARC FILTER DIAGNOSTIC ===")
print(f"Total arcs in distance dict      : {len(d)}")
print(f"Arcs kept (both endpoints in V)  : {len(A)}")
print(f"Arcs skipped (invalid endpoints) : {len(skipped_arcs)}")
if skipped_arcs:
    print("Example skipped arcs:", skipped_arcs[:5])

# -----------------------------
# Adjacency lists
# -----------------------------
OUT = {v: [] for v in V}
IN  = {v: [] for v in V}

for (i, j) in A:
    OUT[i].append(j)
    IN[j].append(i)

# ============================================================
# 3) TIME HELPERS
# ============================================================

def S(i, j, t):
    return service_time.get((i, j, t), 0.0)

def L(i, j, t):
    return d[(i, j)] / v_t[t]

# ============================================================
# 4) COST PARAMETERS
# ============================================================

f = {j: 25.0 for j in J}
c_p = 0.15

c_s = {
    "Car": 12.0,
    "Moped": 6.0,
    "EBikeCart": 4.0,
    "FootBike": 2.0,
}

c_t = {
    "Car": 6.0,
    "Moped": 3.0,
    "EBikeCart": 1.5,
    "FootBike": 0.5,
}

w = 25.0
C_d = 20000
M = 20000

# ============================================================
# 5) MODEL
# ============================================================

m = Model("PostNL_DepotRouting_AggregatedK")

# ============================================================
# 6) VARIABLES
# ============================================================

x = m.addVars(
    [(i, j, t, k) for (i, j) in A for t in T for k in K[t]],
    vtype=GRB.BINARY,
    name="x"
)

y = m.addVars(J, vtype=GRB.BINARY, name="y")
z = m.addVars(I, J, vtype=GRB.BINARY, name="z")

tvar = m.addVars(
    [(i, t, k) for i in I for t in T for k in K[t]],
    lb=0.0,
    name="t"
)

g = m.addVars(
    [(t, k) for t in T for k in K[t]],
    lb=0.0,
    name="g"
)

# ============================================================
# 7) OBJECTIVE
# ============================================================

obj = (
    quicksum(f[j] * y[j] for j in J)
    + 0.56 * quicksum(c_p * D[i] * z[i, j] for i in I for j in J)
    + 0.5 * quicksum(
        c_s[t] * quicksum(x[j, i, t, k] for i in OUT[j])
        for j in J for t in T for k in K[t]
    )
    + quicksum(
        g[t, k] * (c_t[t] + w)
        for t in T for k in K[t]
    )
)

m.setObjective(obj, GRB.MINIMIZE)

# ============================================================
# 8) CONSTRAINTS
# ============================================================

# (6.2) Each route served exactly once
for i in I:
    m.addConstr(
        quicksum(x[i, j, t, k] for j in OUT[i] for t in T for k in K[t]) == 1,
        name=f"assign_{i}"
    )

# (6.3) Route-depot link
for i in I:
    for j in J:
        for t in T:
            for k in K[t]:
                m.addConstr(
                    quicksum(x[i, u, t, k] for u in OUT[i])
                    + quicksum(x[h, j, t, k] for h in IN[j])
                    <= 1 + z[i, j],
                    name=f"link_{i}_{j}_{t}_{k}"
                )

# (6.4) Flow conservation
for t in T:
    for k in K[t]:
        for i in I:
            m.addConstr(
                quicksum(x[i, j, t, k] for j in OUT[i])
                ==
                quicksum(x[j, i, t, k] for j in IN[i]),
                name=f"flow_{i}_{t}_{k}"
            )

# (6.5) One depot start per route index
for t in T:
    for k in K[t]:
        m.addConstr(
            quicksum(x[j, i, t, k] for j in J for i in OUT[j]) <= 1,
            name=f"start_{t}_{k}"
        )

# (6.6) Vehicle capacity
for t in T:
    for k in K[t]:
        m.addConstr(
            quicksum(
                D[i] * quicksum(x[j, i, t, k] for j in IN[i])
                for i in I
            ) <= Q_t[t],
            name=f"cap_{t}_{k}"
        )

# (6.7) Depot capacity
for j in J:
    m.addConstr(
        0.56 * quicksum(D[i] * z[i, j] for i in I) <= C_d * y[j],
        name=f"capDepot_{j}"
    )

# (6.8) Closed depot
for j in J:
    for t in T:
        for k in K[t]:
            m.addConstr(
                quicksum(x[j, i, t, k] for i in OUT[j]) <= y[j],
                name=f"closed_{j}_{t}_{k}"
            )

# (6.9) Time continuity
for t in T:
    for k in K[t]:
        for (i, j) in A:
            if j in I:
                m.addConstr(
                    (tvar[i, t, k] if i in I else 0.0)
                    + S(i, j, t)
                    + L(i, j, t)
                    <= tvar[j, t, k] + M * (1 - x[i, j, t, k]),
                    name=f"time_{i}_{j}_{t}_{k}"
                )

# (6.10) Route duration
for t in T:
    for k in K[t]:
        for (i, j) in A:
            if i in I and j in J:
                m.addConstr(
                    tvar[i, t, k]
                    + S(i, j, t)
                    + L(i, j, t)
                    <= g[t, k] + M * (1 - x[i, j, t, k]),
                    name=f"dur_{i}_{j}_{t}_{k}"
                )

# (6.11) Uptime
for t in T:
    for k in K[t]:
        m.addConstr(
            g[t, k] <= R_t[t],
            name=f"uptime_{t}_{k}"
        )

# ============================================================
# 9) SOLVE
# ============================================================

m.Params.OutputFlag = 1        # ensure output is on
m.Params.DisplayInterval = 1   # print progress every 1 second

m.Params.MIPGap = 0.05
m.Params.TimeLimit = 100
m.optimize()

# ============================================================
# 10) OUTPUT
# ============================================================

if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    print("Open depots:", [j for j in J if y[j].X > 0.5])


if m.Status == GRB.INFEASIBLE:
    print("Model is infeasible — computing IIS...")
    m.computeIIS()
    m.write("infeasible.ilp")