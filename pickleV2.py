# ============================================================
# Data loading + arc construction -> pickle
# ============================================================

import pandas as pd
import pickle

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
PATH_DEPOT_ROUTE = "delivery_times_2028.xlsx"
PATH_ROUTE_ROUTE = "OD_RouteRoute.xlsx"
PATH_DEMAND      = "Demand2028.xlsx"

OUT_FILE = "model_input.pkl"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def to_float(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.replace(".", "").replace(",", ".") if "," in x else x
    return float(x)

def depot_id(raw):
    return f"D{raw}"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df_dr  = pd.read_excel(PATH_DEPOT_ROUTE)
df_rr  = pd.read_excel(PATH_ROUTE_ROUTE)
df_dem = pd.read_excel(PATH_DEMAND)

# ------------------------------------------------------------
# Sets
# ------------------------------------------------------------
I = sorted(df_dem["route_i"].astype(str).unique())
J = sorted(depot_id(j) for j in df_dr["depot_id"].unique())
V = sorted(set(I) | set(J))

T = ["Car", "Moped", "EBikeCart", "FootBike"]

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

# ---- Distances d_ij (km)
d = {}

# Route–Route (already bidirectional)
for _, r in df_rr.iterrows():
    i = str(r["FromRoute"])
    j = str(r["ToRoute"])
    dist_km = to_float(r["Distance"]) / 1000.0
    d[(i, j)] = dist_km

# Route–Depot (bidirectional distance)
for _, r in df_dr.iterrows():
    i = str(r["route_id"])
    j = depot_id(r["depot_id"])
    dist_km = to_float(r["distance_m"]) / 1000.0

    d[(i, j)] = dist_km
    d[(j, i)] = dist_km

# ---- Lead / service times L_(route, depot, t) (hours)
# Interpretation:
# time to deliver all post of route i when supplied from depot j using mode t
L = {}

for _, r in df_dr.iterrows():
    i = str(r["route_id"])
    j = depot_id(r["depot_id"])

    times = {
        "Car":       to_float(r["t_total_min_Car"]),
        "Moped":     to_float(r["t_total_min_Moped"]),
        "EBikeCart": to_float(r["t_total_min_EBikeCart"]),
        "FootBike":  to_float(r["t_total_min_FootBike"]),
    }

    for t, minutes in times.items():
        if minutes is None:
            continue

        hours = minutes / 60.0
        L[(i, j, t)] = hours   # ❗ one-way only

# ---- Demand D_i (delivery bags)
D = {
    str(r["route_i"]): int(r["DeliveryBags"])
    for _, r in df_dem.iterrows()
}

# ------------------------------------------------------------
# Bundle for pickle
# ------------------------------------------------------------
model_data = {
    "sets": {
        "I": I,
        "J": J,
        "V": V,
        "T": T,
    },
    "parameters": {
        "d": d,
        "L": L,
        "D": D,
    }
}

# ------------------------------------------------------------
# Save pickle
# ------------------------------------------------------------
with open(OUT_FILE, "wb") as f:
    pickle.dump(model_data, f)

print(f"Pickle written to {OUT_FILE}")
print(f"|I|={len(I)}, |J|={len(J)}, |V|={len(V)}, |arcs|={len(d)}, |L|={len(L)}")
