# ============================================================
# MILP Depot + Routing model (PICKLE-BASED, VEHICLE TYPES ONLY)
# ============================================================

import pickle
from gurobipy import Model, GRB, quicksum

# ============================================================
# 0) LOAD DATA FROM PICKLE
# ============================================================

with open("model_input.pkl", "rb") as f:
    model_data = pickle.load(f)

# -----------------------------
# Sets
# -----------------------------
I = model_data["sets"]["I"]          # routes
J = model_data["sets"]["J"]          # depots
V = model_data["sets"]["V"]          # nodes
T = model_data["sets"]["T"]          # vehicle types (4 only)

# -----------------------------
# Parameters
# -----------------------------
d = model_data["parameters"]["d"]                    # distances (km)
D = model_data["parameters"]["D"]                    # demand (bags)
service_time = model_data["parameters"]["service_time"]  # (i,j,t) → hours

# ============================================================
# 1) VEHICLE AVAILABILITY (AGGREGATED)
# ============================================================

# -----------------------------
# Vehicle velocities (km/hour)
# -----------------------------
v_t = {
    "Car": 40.0,
    "Moped": 30.0,
    "EBikeCart": 20.0,
    "FootBike": 15.0,
}

N_VEH = {
    "Car": 1000,
    "Moped": 1000,
    "EBikeCart": 1000,
    "FootBike": 1000,
}
# ============================================================
# 2) ARC SET (FILTERED) + DIAGNOSTICS
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
    """Service time in hours (route → depot only)."""
    return service_time.get((i, j, t), 0.0)

def L(i, j, t):
    """Lead (travel) time in hours."""
    return d[(i, j)] / v_t[t]

# ============================================================
# 4) COST & CAPACITY PARAMETERS (REALISTIC)
# ============================================================

# Fixed depot opening cost (€/day)
f = {j: 25.0 for j in J}

# Process cost per delivery bag (€/bag)
c_p = 0.15

# Vehicle storage cost (€/vehicle/day)
c_s = {
    "Car": 12.0,
    "Moped": 6.0,
    "EBikeCart": 4.0,
    "FootBike": 2.0,
}

# Vehicle operational cost (€/hour, excl. labor)
c_t = {
    "Car": 6.0,
    "Moped": 3.0,
    "EBikeCart": 1.5,
    "FootBike": 0.5,
}

# Mail carrier wage (€/hour)
w = 25.0

# Depot capacity (bags/day)
C_d = 20000

# Vehicle capacity (bags per vehicle)
Q_t = {
    "Car": 20,
    "Moped": 6,
    "EBikeCart": 6,
    "FootBike": 3,
}

# Maximum uptime per vehicle (hours/day)
R_t = {
    "Car": 8.0,
    "Moped": 7.5,
    "EBikeCart": 7.0,
    "FootBike": 6.0,
}

# Big-M (now much tighter)
M = 20000


# -----------------------------
# Sanity check (development only)
# -----------------------------
assert all(t in Q_t and t in R_t and t in v_t for t in T), \
    "Missing vehicle parameters for at least one vehicle type" 

# ============================================================
# 5) MODEL
# ============================================================

m = Model("PostNL_DepotRouting_Types")

# ============================================================
# 6) VARIABLES
# ============================================================

# Arc usage by vehicle type
x = m.addVars(A, T, vtype=GRB.BINARY, name="x")

# Depot open
y = m.addVars(J, vtype=GRB.BINARY, name="y")

# Route → depot assignment
z = m.addVars(I, J, vtype=GRB.BINARY, name="z")

# Arrival time at route
tvar = m.addVars(I, T, lb=0.0, name="t")

# Total operating time per vehicle type
g = m.addVars(T, lb=0.0, name="g")

# ============================================================
# 7) OBJECTIVE FUNCTION
# ============================================================

obj_fixed = quicksum(f[j] * y[j] for j in J)

obj_process = 0.56 * quicksum(
    c_p * D[i] * z[i, j]
    for i in I for j in J
)

obj_storage = 0.5 * quicksum(
    c_s[t] *
    quicksum(x[j, i, t] for j in J for i in OUT[j])
    for t in T
)

obj_route = quicksum(
    g[t] * (c_t[t] + w)
    for t in T
)

m.setObjective(
    obj_fixed + obj_process + obj_storage + obj_route,
    GRB.MINIMIZE
)

# ============================================================
# 8) CONSTRAINTS
# ============================================================

# (6.2) Each route served exactly once (by some vehicle type)
for i in I:
    m.addConstr(
        quicksum(x[i, j, t] for j in OUT[i] for t in T) == 1,
        name=f"assign_{i}"
    )

# (6.7) Depot capacity
for j in J:
    m.addConstr(
        0.56 * quicksum(D[i] * z[i, j] for i in I) <= C_d * y[j],
        name=f"cap_depot_{j}"
    )

# Route assigned to exactly one depot
for i in I:
    m.addConstr(
        quicksum(z[i, j] for j in J) == 1,
        name=f"assignDepot_{i}"
    )

# (6.8) Closed depot
for j in J:
    for t in T:
        m.addConstr(
            quicksum(x[j, i, t] for i in OUT[j]) <= y[j],
            name=f"closed_{j}_{t}"
        )

# (6.6) Vehicle capacity (per vehicle type)
for t in T:
    m.addConstr(
        quicksum(
            D[i] * quicksum(x[j, i, t] for j in IN[i])
            for i in I
        ) <= Q_t[t] * N_VEH[t],
        name=f"veh_cap_{t}"
    )


# (6.9) Time continuity WITH lead time
for t in T:
    for (i, j) in A:
        if j in I:
            m.addConstr(
                (tvar[i, t] if i in I else 0.0)
                + S(i, j, t)
                + L(i, j, t)
                <= tvar[j, t] + M * (1 - x[i, j, t]),
                name=f"time_{i}_{j}_{t}"
            )

# (6.10) Global route duration WITH lead time
for t in T:
    for (i, j) in A:
        if i in I and j in J:
            m.addConstr(
                tvar[i, t]
                + S(i, j, t)
                + L(i, j, t)
                <= g[t] + M * (1 - x[i, j, t]),
                name=f"dur_{i}_{j}_{t}"
            )

# (6.11) Vehicle uptime (per vehicle type)
for t in T:
    m.addConstr(
        g[t] <= R_t[t] * N_VEH[t],
        name=f"uptime_{t}"
    )


# ============================================================
# 9) SOLVE (MUCH SMALLER CUTOFF)
# ============================================================

m.Params.MIPGap = 0.05
m.Params.TimeLimit = 100    # 5 minutes
m.optimize()

# ============================================================
# 10) OUTPUT
# ============================================================

if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    open_depots = [j for j in J if y[j].X > 0.5]
    print("Open depots:", open_depots)

if m.Status == GRB.INFEASIBLE:
    print("Model is infeasible — computing IIS...")
    m.computeIIS()
    m.write("infeasible.ilp")
