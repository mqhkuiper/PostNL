# ============================================================
# MILP Depot + Routing model (PICKLE-BASED)
# ============================================================

import pickle
from gurobipy import Model, GRB, quicksum

# ============================================================
# 0) LOAD DATA FROM PICKLE (ALIGNED WITH CREATION SCRIPT)
# ============================================================

import pickle

with open("model_input.pkl", "rb") as f:
    model_data = pickle.load(f)

# -----------------------------
# Sets
# -----------------------------
I = model_data["sets"]["I"]          # routes
J = model_data["sets"]["J"]          # depots
V = model_data["sets"]["V"]          # nodes = I ∪ J
T = model_data["sets"]["T"]          # vehicle types

# -----------------------------
# Parameters
# -----------------------------
d = model_data["parameters"]["d"]                    # distances (km)
D = model_data["parameters"]["D"]                    # demand per route
service_time = model_data["parameters"]["service_time"]  # (route, depot, t) → hours


# -----------------------------
# Vehicles
# -----------------------------
N_VEH = {
    "Car": 50,
    "Moped": 200,
    "EBikeCart": 200,
    "FootBike": 400,
}

T = [(t, n) for t in T for n in range(1, N_VEH[t] + 1)]

def mode_of(k):
    return k[0]

# -----------------------------
# Vehicle velocities (km/hour)
# -----------------------------
v_t = {
    "Car": 40.0,
    "Moped": 30.0,
    "EBikeCart": 20.0,
    "FootBike": 15.0,
}

# -----------------------------
# Arc set derived from distances (FILTERED)
# -----------------------------
A = []
skipped_arcs = []

for (i, j) in d.keys():
    if i in V and j in V:
        A.append((i, j))
    else:
        skipped_arcs.append((i, j))

# Diagnostic output
print("\n=== ARC FILTER DIAGNOSTIC ===")
print(f"Total arcs in distance dict      : {len(d)}")
print(f"Arcs kept (both endpoints in V)  : {len(A)}")
print(f"Arcs skipped (invalid endpoints) : {len(skipped_arcs)}")

# Optional: show a few examples
if skipped_arcs:
    print("Example skipped arcs:", skipped_arcs[:5])

# -----------------------------
# Adjacency lists (SAFE)
# -----------------------------
OUT = {v: [] for v in V}
IN  = {v: [] for v in V}

for (i, j) in A:
    OUT[i].append(j)
    IN[j].append(i)
# -----------------------------
# Helper: service time lookup
# -----------------------------
def S(i, j, t):
    """
    Service time in hours.
    Only defined for route → depot arcs.
    """
    if (i, j, t) in service_time:
        return service_time[(i, j, t)]
    return 0.0

def L(i, j, t):
    """
    Lead (travel) time in hours from i to j using vehicle type t.
    """
    return d[(i, j)] / v_t[t]

# ============================================================
# 3) COST & CAPACITY PARAMETERS (PLACEHOLDERS)
# ============================================================

# Fixed depot opening cost
f = {j: 0.0 for j in J}                         # TODO €/depot

# Process cost per delivery bag
c_p = 0.0                                       # TODO €/bag

# Vehicle storage cost per vehicle type
c_s = {t: 0.0 for t in T}                       # TODO €/vehicle

# Vehicle operational cost per hour
c_t = {t: 0.0 for t in T}                       # TODO €/hour

# Mail carrier wage per hour
w = 0.0                                        # TODO €/hour 

# Capacities
C_d = 1e9                                      # depot capacity (bags)
Q_t = {t: 999 for t in T}                      # vehicle capacity (bags)

# Maximum vehicle uptime
R_t = {t: 8.0 for t in T}                      # hours

# Big-M
M = 1e5



# ============================================================
# 4) MODEL
# ============================================================

m = Model("PostNL_DepotRouting")

# ============================================================
# 5) VARIABLES
# ============================================================

# Arc usage
x = m.addVars(
    [(i, j, k) for (i, j) in A for k in T],
    vtype=GRB.BINARY,
    name="x"
)

# Depot open
y = m.addVars(J, vtype=GRB.BINARY, name="y")

# Route → depot assignment
z = m.addVars(I, J, vtype=GRB.BINARY, name="z")

# Arrival time at route
t = m.addVars(I, T, lb=0.0, name="t")

# Global route duration
g = m.addVars(T, lb=0.0, name="g")

# ============================================================
# 6) OBJECTIVE FUNCTION  (Eq. 6.1)
# ============================================================

obj_fixed = quicksum(f[j] * y[j] for j in J)

obj_process = 0.56 * quicksum(
    c_p * D[i] * z[i, j]
    for i in I for j in J
)

obj_storage = 0.5 * quicksum(
    c_s[mode_of(k)] *
    quicksum(x[j, i, k] for i in OUT[j] if (j, i, k) in x)
    for j in J for k in T
)

obj_route = quicksum(
    g[k] * (c_t[mode_of(k)] + w)
    for k in T
)

m.setObjective(
    obj_fixed + obj_process + obj_storage + obj_route,
    GRB.MINIMIZE
)

# ============================================================
# 7) CONSTRAINTS
# ============================================================

# (6.2) Assignment constraint
for i in I:
    m.addConstr(
        quicksum(x[i, j, k] for j in OUT[i] for k in T if (i, j, k) in x) == 1,
        name=f"assign_{i}"
    )

# (6.4) Flow conservation
for k in T:
    for i in I:
        m.addConstr(
            quicksum(x[i, j, k] for j in OUT[i] if (i, j, k) in x)
            ==
            quicksum(x[j, i, k] for j in IN[i] if (j, i, k) in x),
            name=f"flow_{i}_{k}"
        )

# (6.5) One depot start per vehicle
for k in T:
    m.addConstr(
        quicksum(x[j, i, k] for j in J for i in OUT[j] if (j, i, k) in x) <= 1,
        name=f"start_{k}"
    )

# (6.7) Depot capacity
for j in J:
    m.addConstr(
        0.56 * quicksum(D[i] * z[i, j] for i in I) <= C_d * y[j],
        name=f"cap_depot_{j}"
    )

# (6.8) Closed depot constraint
for j in J:
    for k in T:
        m.addConstr(
            quicksum(x[j, i, k] for i in OUT[j] if (j, i, k) in x) <= y[j],
            name=f"closed_{j}_{k}"
        )

# Route assigned to exactly one depot
for i in I:
    m.addConstr(
        quicksum(z[i, j] for j in J) == 1,
        name=f"assignDepot_{i}"
    )

# (6.6) Vehicle capacity
for k in T:
    m.addConstr(
        quicksum(
            D[i] *
            quicksum(x[j, i, k] for j in IN[i] if (j, i, k) in x)
            for i in I
        ) <= Q_t[mode_of(k)],
        name=f"cap_vehicle_{k}"
    )

# (6.9) Time continuity WITH lead time
for k in T:
    tmode = mode_of(k)
    for (i, j) in A:
        if j in I and (i, j) in d:
            service = 0.0 if i in J else S_arc.get((i, j, tmode), 0.0)
            lead_time = d[(i, j)] / v_t[tmode]

            m.addConstr(
                (t[i, k] if i in I else 0.0)
                + service
                + lead_time
                <= t[j, k] + M * (1 - x[i, j, k]),
                name=f"time_{i}_{j}_{k}"
            )

# (6.10) Global route duration WITH lead time
for k in T:
    tmode = mode_of(k)
    for (i, j) in A:
        if i in I and j in J and (i, j) in d:
            service = S_arc.get((i, j, tmode), 0.0)
            lead_time = d[(i, j)] / v_t[tmode]

            m.addConstr(
                t[i, k]
                + service
                + lead_time
                <= g[k] + M * (1 - x[i, j, k]),
                name=f"dur_{i}_{j}_{k}"
            )

# (6.11) Vehicle uptime
for k in T:
    m.addConstr(
        g[k] <= R_t[mode_of(k)],
        name=f"uptime_{k}"
    )

# ============================================================
# 8) SOLVE
# ============================================================

m.Params.MIPGap = 0.01
m.Params.TimeLimit = 3600
m.optimize()

# ============================================================
# 9) OUTPUT (BASIC)
# ============================================================

if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    open_depots = [j for j in J if y[j].X > 0.5]
    print("Open depots:", open_depots)
