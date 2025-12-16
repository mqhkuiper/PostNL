# ============================================================
# MILP Depot + Routing model 
# ============================================================
import pandas as pd
from gurobipy import Model, GRB, quicksum


# 0) File Paths 

PATH_DEPOT_ROUTE = "delivery_times_2028.xlsx"   # columns: route_id, depot_id, distance_m, t_total_min_Car,	t_total_min_Moped, t_total_min_EBikeCart, t_total_min_FootBike
PATH_ROUTE_ROUTE = "OD_RouteRoute.xlsx"         # columns: FromRoute, ToRoute, Distance
PATH_DEMAND      = "Demand2028.xlsx"            # columns: route_i, DemandDaily_Letters_2028, DemandDaily_LBP_2028, DeliveryBags
 

# 1) Data helper

def to_float(x):
    """Conversion for Excel values that use comma decimals."""
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.replace(".", "").replace(",", ".") if "," in x else x
    return float(x)

def depot_id(raw):
    """Prefix depot IDs so they never collide with route IDs."""
    return f"D{raw}"


# 2) LOAD DATA

df_dr = pd.read_excel(PATH_DEPOT_ROUTE)
df_rr = pd.read_excel(PATH_ROUTE_ROUTE)
df_dem = pd.read_excel(PATH_DEMAND)


# Clean/standardize IDs
df_dr["route"] = df_dr["route_i"].astype(str)
df_dr["depot"] = df_dr["depot_i"].apply(lambda x: depot_id(str(int(x)) if pd.notna(x) else x))

df_rr["i"] = df_rr["FromRoute"].astype(str)
df_rr["j"] = df_rr["ToRoute"].astype(str)

df_dem["route"] = df_dem["bestelloop"].astype(str)


# 3) SETS
I = sorted(df_dem["route"].unique().tolist())                     # customers/routes
J = sorted(df_dr["depot"].unique().tolist())                      # candidate depots
V = I + J

# transport modes (match your Excel column suffixes)
T = ["Car", "Moped", "EBikeCart", "FootBike"]

# Create individual vehicles K (EDIT counts!)
N_VEH = {"Car": 50, "Moped": 200, "EBikeCart": 200, "FootBike": 400}
K = [(t, n) for t in T for n in range(1, N_VEH[t] + 1)]           # vehicle = (mode, idx)

def mode_of(k):
    return k[0]

# -----------------------------
# 4) PARAMETERS (from Excel)
# -----------------------------
# Demand D_i (delivery bags)
D = dict(zip(df_dem["route"], df_dem["Tassen"].apply(to_float)))

# Distances / travel times for arcs (i,j)
dist = {}          # km
travel = {}        # hours per mode: travel[(i,j,t)] = hours

# ---- depot <-> route arcs from df_dr
# distance_m is meters in your screenshot; convert to km
for _, r in df_dr.iterrows():
    i = r["depot"]       # depot node
    j = r["route"]       # route node
    d_km = to_float(r["distance_m"]) / 1000.0
    dist[(i, j)] = d_km
    dist[(j, i)] = d_km  # assume symmetric (adjust if you have directed times)

    # minutes -> hours
    for t in T:
        col = f"t_total_min_{t}"
        if col not in df_dr.columns:
            # your screenshot shows names like t_total_min_Ca... (truncated in Excel view)
            # TODO: rename your columns OR map here.
            continue
        m = to_float(r[col])
        if m is None:
            continue
        travel[(i, j, t)] = m / 60.0
        travel[(j, i, t)] = m / 60.0

# ---- route -> route arcs from df_rr
for _, r in df_rr.iterrows():
    i = r["i"]
    j = r["j"]
    d_km = to_float(r["Distance"]) / 1000.0 if to_float(r["Distance"]) > 100 else to_float(r["Distance"]) / 1000.0
    # NOTE: your Route-Route "Distance" looks like meters already (e.g., 4873).
    dist[(i, j)] = d_km

    for t in T:
        col = f"t_travel_min_{t}"
        if col not in df_rr.columns:
            continue
        m = to_float(r[col])
        if m is None:
            continue
        travel[(i, j, t)] = m / 60.0

# -----------------------------
# 5) BUILD ARC SET A (only arcs we actually have data for)
# -----------------------------
A = set()
A |= set((a, b) for (a, b) in dist.keys() if a in V and b in V and a != b)

# Convenience adjacency lists
OUT = {v: [] for v in V}
IN  = {v: [] for v in V}
for (i, j) in A:
    OUT[i].append(j)
    IN[j].append(i)

# -----------------------------
# 6) OTHER PARAMETERS (FILL THESE!)
# -----------------------------
# Depot opening costs
f = {j: 0.0 for j in J}               # TODO fill from your cost model

# process cost per bag
c_v = 0.0                              # TODO

# vehicle storage cost per vehicle used at depot
c_cpl = {"Car": 0.0, "Moped": 0.0, "EBikeCart": 0.0, "FootBike": 0.0}  # TODO

# vehicle cost per hour
c_k = {"Car": 0.0, "Moped": 0.0, "EBikeCart": 0.0, "FootBike": 0.0}    # TODO

# wage per hour
w = 0.0                                # TODO

# capacities
C_d = 1e9                              # TODO depot capacity in bags
Q_t = {"Car": 999, "Moped": 999, "EBikeCart": 999, "FootBike": 999}    # TODO

# max uptime
R_t = {"Car": 8.0, "Moped": 8.0, "EBikeCart": 8.0, "FootBike": 8.0}    # TODO (hours)

# service time S_{ik} (hours) = duration of delivery route i with vehicle mode t
# TODO: load from your route duration table; for now set 0 so the model runs.
S = {(i, k): 0.0 for i in I for k in K}

M = 1e5

# ============================================================
# 7) GUROBI MODEL
# ============================================================
m = Model("DepotRouting")

# -----------------------------
# Variables
# -----------------------------
# x[i,j,k] only where (i,j) in A
X_index = [(i, j, k) for (i, j) in A for k in K]
x = m.addVars(X_index, vtype=GRB.BINARY, name="x")

y = m.addVars(J, vtype=GRB.BINARY, name="y")                 # depot open
z = m.addVars(I, J, vtype=GRB.BINARY, name="z")              # route assigned to depot
tvar = m.addVars(I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="t")  # arrival time at route i
g = m.addVars(K, lb=0.0, vtype=GRB.CONTINUOUS, name="g")     # global route duration

# -----------------------------
# Objective (your structure, but implemented safely)
# -----------------------------
obj_fixed = quicksum(f[j] * y[j] for j in J)

# process cost (your latex missed the sum over i)
obj_process = 0.5 * quicksum(c_v * D[i] for i in I)

# coupling/storage cost: count vehicles that leave a depot at least once
# Here: sum_{i} x[j,i,k] is 1 if vehicle k starts from depot j (because of depot-start <= 1)
obj_cpl = 0.5 * quicksum(c_cpl[mode_of(k)] * quicksum(x[j, i, k] for i in OUT[j] if (j, i) in A)
                         for j in J for k in K)

obj_route = quicksum(g[k] * (c_k[mode_of(k)] + w) for k in K)

m.setObjective(obj_fixed + obj_process + obj_cpl + obj_route, GRB.MINIMIZE)

# ============================================================
# 8) CONSTRAINTS
# ============================================================

# (1) Assignment constraint: each route i has exactly one outgoing arc over all vehicles
for i in I:
    m.addConstr(
        quicksum(x[i, j, k] for j in OUT[i] for k in K if (i, j, k) in x) == 1,
        name=f"assign_out_{i}"
    )

# (2) Continuity: flow conservation for each vehicle at each route node
for k in K:
    for i in I:
        m.addConstr(
            quicksum(x[i, j, k] for j in OUT[i] if (i, j, k) in x) ==
            quicksum(x[j, i, k] for j in IN[i]  if (j, i, k) in x),
            name=f"flow_{i}_{k}"
        )

# (3) Depot start: each vehicle starts from at most one depot
for k in K:
    m.addConstr(
        quicksum(x[j, i, k] for j in J for i in OUT[j] if (j, i, k) in x) <= 1,
        name=f"depot_start_{k}"
    )

# (4) Closed depot: can only leave depot if open
for j in J:
    for k in K:
        m.addConstr(
            quicksum(x[j, i, k] for i in OUT[j] if (j, i, k) in x) <= y[j],
            name=f"closed_{j}_{k}"
        )

# (5) Depot capacity: sum of demands assigned to depot
for j in J:
    m.addConstr(
        quicksum(D[i] * z[i, j] for i in I) <= C_d * y[j],
        name=f"depot_cap_{j}"
    )

# (6) Vehicle capacity: served demand by vehicle k <= Q_t
for k in K:
    tmode = mode_of(k)
    m.addConstr(
        quicksum(D[i] * quicksum(x[j, i, k] for j in IN[i] if (j, i, k) in x) for i in I) <= Q_t[tmode],
        name=f"veh_cap_{k}"
    )

# (7) Link route to depot assignment (simple + strong version)
# If vehicle k leaves depot j to first customer i, then i must be assigned to j
for j in J:
    for i in I:
        for k in K:
            if (j, i, k) in x:
                m.addConstr(x[j, i, k] <= z[i, j], name=f"link_start_{i}_{j}_{k}")

# Each route assigned to exactly one depot (recommended)
for i in I:
    m.addConstr(quicksum(z[i, j] for j in J) == 1, name=f"assign_depot_{i}")

# (8) Time continuity (MTZ-style) for arcs into routes
# t_i + service(i) + travel(i->j) <= t_j + M(1-x_ijk)
for k in K:
    tmode = mode_of(k)
    for (i, j) in A:
        if j in I and (i, j, tmode) in travel and (i, j, k) in x:
            s_i = 0.0 if i in J else S[(i, k)]
            m.addConstr(
                (tvar[i, k] if i in I else 0.0) + s_i + travel[(i, j, tmode)]
                <= tvar[j, k] + M * (1 - x[i, j, k]),
                name=f"time_{i}_{j}_{k}"
            )

# (9) Route duration: if vehicle goes from a route i back to depot j, bound g_k
for k in K:
    tmode = mode_of(k)
    for (i, j) in A:
        if i in I and j in J and (i, j, tmode) in travel and (i, j, k) in x:
            m.addConstr(
                tvar[i, k] + S[(i, k)] + travel[(i, j, tmode)] - M * (1 - x[i, j, k]) <= g[k],
                name=f"dur_{i}_{j}_{k}"
            )

# (10) Uptime
for k in K:
    m.addConstr(g[k] <= R_t[mode_of(k)], name=f"uptime_{k}")

# ============================================================
# 9) SOLVE (basic)
# ============================================================
m.Params.MIPGap = 0.01
m.Params.TimeLimit = 3600

m.optimize()

# After solve: read solution (example)
if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
    open_depots = [j for j in J if y[j].X > 0.5]
    print("Open depots:", open_depots)
