import pickle
from gurobipy import Model, GRB, quicksum

# ============================================================
# 0) LOAD DATA
# ============================================================

with open("model_input.pkl", "rb") as f:
    model_data = pickle.load(f)

I = model_data["sets"]["I"]          # routes
J = model_data["sets"]["J"]          # depots
T = model_data["sets"]["T"]          # modes

d = model_data["parameters"]["d"]              # distance (km)
D = model_data["parameters"]["D"]              # demand (bags)
S = model_data["parameters"]["service_time"]   # service time (hours)

# ============================================================
# 1) PARAMETERS
# ============================================================

# Speeds (km/h)
v_t = {
    "Car": 40.0,
    "Moped": 30.0,
    "EBikeCart": 20.0,
    "FootBike": 15.0,
}

# Vehicle capacity (bags)
Q_t = {
    "Car": 20,
    "Moped": 6,
    "EBikeCart": 6,
    "FootBike": 3,
}

# Vehicle maximum uptime (hours)
R_t = {
    "Car": 8.0,
    "Moped": 7.5,
    "EBikeCart": 7.0,
    "FootBike": 6.0,
}

# Fixed per-route cost per mode
c_s = {
    "Car": 12.0,
    "Moped": 6.0,
    "EBikeCart": 4.0,
    "FootBike": 2.0,
}

# Vehicle operating cost €/hour
c_t = {
    "Car": 6.0,
    "Moped": 3.0,
    "EBikeCart": 1.5,
    "FootBike": 0.5,
}

w   = 25.0     # wage €/hour
c_p = 0.15     # per-bag handling cost
f   = {j: 25.0 for j in J}   # depot fixed cost
C_d = 20000    # depot capacity (bags)

# ============================================================
# 2) PRECOMPUTE ASSIGNMENT COSTS
# ============================================================

C = {}   # (i, j, t) -> cost
missing_distance = 0

for (i, j, t), service_time in S.items():

    if (i, j) not in d:
        missing_distance += 1
        continue

    travel_time = d[(i, j)] / v_t[t]

    travel_cost  = travel_time * (c_t[t] + w)
    service_cost = service_time * (c_t[t] + w)

    C[i, j, t] = (
        travel_cost
        + service_cost
        + c_s[t]
        + c_p * D[i]
    )

print(f"Computed costs for {len(C)} (route, depot, mode) triples")
print(f"Skipped due to missing distance: {missing_distance}")

# ------------------------------------------------------------
# FIX: Remove routes without any assignment options
# ------------------------------------------------------------
routes_with_assignments = {i for (i, j, t) in C}
routes_without_assignments = set(I) - routes_with_assignments

if routes_without_assignments:
    print(f"WARNING: Removing {len(routes_without_assignments)} routes with no assignment options")
    print("Example routes:", list(routes_without_assignments)[:5])

I = sorted(routes_with_assignments)

# ============================================================
# 3) MODEL
# ============================================================

m = Model("PostNL_DepotLocation_AggregatedVehicles")

# ============================================================
# 4) VARIABLES
# ============================================================

# Depot open
y = m.addVars(J, vtype=GRB.BINARY, name="openDepot")

# Route assignment (route, depot, mode)
z = m.addVars(
    C.keys(),
    vtype=GRB.BINARY,
    name="assign"
)

# Number of vehicles per depot and type
n = m.addVars(
    [(j, t) for j in J for t in T],
    vtype=GRB.INTEGER,
    lb=0,
    name="numVehicles"
)

# ============================================================
# 5) OBJECTIVE
# ============================================================

m.setObjective(
    # depot opening
    quicksum(f[j] * y[j] for j in J)

    # route assignment costs
    + quicksum(C[i, j, t] * z[i, j, t] for (i, j, t) in C)

    # vehicle storage / ownership costs
    + quicksum(c_t[t] * n[j, t] for j in J for t in T),
    GRB.MINIMIZE
)

# ============================================================
# 6) CONSTRAINTS
# ============================================================

# ------------------------------------------------------------
# Each route assigned exactly once
# ------------------------------------------------------------
for i in I:
    m.addConstr(
        quicksum(z[i, j, t] for (ii, j, t) in C if ii == i) == 1,
        name=f"assign_{i}"
    )

# ------------------------------------------------------------
# Depot capacity
# ------------------------------------------------------------
for j in J:
    m.addConstr(
        quicksum(D[i] * z[i, j, t] for (i, jj, t) in C if jj == j)
        <= C_d * y[j],
        name=f"capDepot_{j}"
    )

# ------------------------------------------------------------
# Cannot assign routes or vehicles to closed depots
# ------------------------------------------------------------
for (i, j, t) in C:
    m.addConstr(z[i, j, t] <= y[j])

for j in J:
    for t in T:
        m.addConstr(n[j, t] <= 10_000 * y[j])

# ------------------------------------------------------------
# Aggregated vehicle capacity constraints
# ------------------------------------------------------------
for j in J:
    for t in T:
        m.addConstr(
            quicksum(
                D[i] * z[i, j, t]
                for (i, jj, tt) in C if jj == j and tt == t
            )
            <= Q_t[t] * n[j, t],
            name=f"vehCap_{j}_{t}"
        )

# ------------------------------------------------------------
# Aggregated vehicle uptime constraints
# ------------------------------------------------------------
for j in J:
    for t in T:
        m.addConstr(
            quicksum(
                (S[i, j, t] + d[(i, j)] / v_t[t]) * z[i, j, t]
                for (i, jj, tt) in C if jj == j and tt == t
            )
            <= R_t[t] * n[j, t],
            name=f"vehTime_{j}_{t}"
        )

# ============================================================
# 7) SOLVE
# ============================================================

m.Params.OutputFlag = 1
m.Params.MIPGap = 0.05
m.Params.TimeLimit = 300

m.optimize()

# ============================================================
# 8) OUTPUT
# ============================================================

if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:

    print("\nOpen depots:")
    print([j for j in J if y[j].X > 0.5])

    print("\nVehicles per depot and type (nonzero only):")
    for j in J:
        for t in T:
            if n[j, t].X > 0.5:
                print(f"Depot {j}: {int(round(n[j, t].X))} × {t}")

    print("\nSample assignments:")
    shown = 0
    for (i, j, t) in C:
        if z[i, j, t].X > 0.5:
            print(f"Route {i} → Depot {j} via {t}")
            shown += 1
            if shown >= 10:
                break

if m.Status == GRB.INFEASIBLE:
    print("Model infeasible — computing IIS")
    m.computeIIS()
    m.write("infeasible.ilp")
