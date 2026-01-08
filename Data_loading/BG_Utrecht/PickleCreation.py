# ============================================================
# Data loading + arc construction -> pickle (WITH XY ROUTES)
# routes without XY code are dropped consistently
# also drops routes without demand, and filters all arcs accordingly
# ============================================================

import pandas as pd
import pickle
from pathlib import Path

# ------------------------------------------------------------
# file paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PATH_DEPOT_ROUTE = BASE_DIR / "delivery_times_2028.xlsx"
PATH_ROUTE_ROUTE = BASE_DIR / "OD_RouteRoute.xlsx"
PATH_DEMAND      = BASE_DIR / "Demand2028.xlsx"
PATH_XY          = BASE_DIR / "XYroutes.xlsx"

OUT_FILE = BASE_DIR / "model_inputBGU.pkl"

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def to_float(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip()
        x = x.replace(".", "").replace(",", ".") if "," in x else x
    try:
        return float(x)
    except Exception:
        return None

def clean_id(x):
    if pd.isna(x):
        return None
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def route_id(x):
    return clean_id(x)

def depot_id(x):
    s = clean_id(x)
    return None if s is None else f"D{s}"

# ------------------------------------------------------------
# load data
# ------------------------------------------------------------

df_dr  = pd.read_excel(PATH_DEPOT_ROUTE)
df_rr  = pd.read_excel(PATH_ROUTE_ROUTE)
df_dem = pd.read_excel(PATH_DEMAND)
df_xy  = pd.read_excel(PATH_XY)

# ------------------------------------------------------------
# clean/standardize key columns
# ------------------------------------------------------------

# route-depot table
df_dr["route_clean"] = df_dr["route_id"].apply(route_id)
df_dr["depot_clean"] = df_dr["depot_id"].apply(depot_id)

# route-route od table
df_rr["from_clean"] = df_rr["FromRoute"].apply(route_id)
df_rr["to_clean"]   = df_rr["ToRoute"].apply(route_id)

# demand table
df_dem["route_clean"] = df_dem["route_i"].apply(route_id)

# xy table
df_xy["route_clean"] = df_xy["bestelloop"].apply(route_id)
df_xy["xy_code"] = df_xy["xy_code"].astype(str).str.strip().str.upper()

# drop rows with missing keys
df_dr  = df_dr.dropna(subset=["route_clean", "depot_clean"]).copy()
df_rr  = df_rr.dropna(subset=["from_clean", "to_clean"]).copy()
df_dem = df_dem.dropna(subset=["route_clean"]).copy()
df_xy  = df_xy.dropna(subset=["route_clean", "xy_code"]).copy()

# keep only valid xy codes
df_xy = df_xy[df_xy["xy_code"].isin(["X", "Y"])].copy()

# ------------------------------------------------------------
# build xy mapping (only valid codes)
# if duplicates exist, last one wins (you can change to raise if needed)
# ------------------------------------------------------------

xy_all = {}
for _, r in df_xy.iterrows():
    xy_all[r["route_clean"]] = r["xy_code"]

print(f"[info] xy rows (valid X/Y): {len(df_xy)}")
print(f"[info] unique routes with xy: {len(xy_all)}")

# ------------------------------------------------------------
# determine active routes:
# must have demand + xy code
# ------------------------------------------------------------

routes_with_demand = set(df_dem["route_clean"].unique())
routes_with_xy = set(xy_all.keys())

active_routes = routes_with_demand & routes_with_xy

print(f"[info] routes with demand      : {len(routes_with_demand)}")
print(f"[info] routes with xy          : {len(routes_with_xy)}")
print(f"[cleanup] active routes (I)    : {len(active_routes)}")
print(f"[cleanup] dropped (no xy)      : {len(routes_with_demand - routes_with_xy)}")
print(f"[cleanup] dropped (no demand)  : {len(routes_with_xy - routes_with_demand)}")

# filter demand to active routes
df_dem = df_dem[df_dem["route_clean"].isin(active_routes)].copy()

# filter route-depot to active routes only
df_dr = df_dr[df_dr["route_clean"].isin(active_routes)].copy()

# filter route-route to active routes on both ends
df_rr = df_rr[
    df_rr["from_clean"].isin(active_routes) &
    df_rr["to_clean"].isin(active_routes)
].copy()

# ------------------------------------------------------------
# sets
# ------------------------------------------------------------

I = sorted(df_dem["route_clean"].unique())
J = sorted(df_dr["depot_clean"].unique())
V = sorted(set(I) | set(J))
T = ["Car", "Moped", "EBikeCart", "FootBike"]

print(f"[sets] |I|={len(I)}, |J|={len(J)}, |V|={len(V)}, |T|={len(T)}")

# keep xy only for routes that are actually in I (so no extras in the pickle)
xy = {i: xy_all[i] for i in I}

# ------------------------------------------------------------
# distances d_ij (km)
# only between nodes in V = I ∪ J (guaranteed by construction, but we also check)
# ------------------------------------------------------------

d = {}

# route–route distances
missing_rr = 0
for _, r in df_rr.iterrows():
    i, j = r["from_clean"], r["to_clean"]
    dist_m = to_float(r["Distance"])
    if dist_m is None:
        missing_rr += 1
        continue
    d[(i, j)] = dist_m / 1000.0

# route–depot distances (bidirectional)
missing_dr = 0
for _, r in df_dr.iterrows():
    i, j = r["route_clean"], r["depot_clean"]
    dist_m = to_float(r["distance_m"])
    if dist_m is None:
        missing_dr += 1
        continue
    dist_km = dist_m / 1000.0
    d[(i, j)] = dist_km
    d[(j, i)] = dist_km

if missing_rr > 0:
    raise ValueError(f"missing/invalid route-route distances in {missing_rr} rows")
if missing_dr > 0:
    raise ValueError(f"missing/invalid route-depot distances in {missing_dr} rows")

# hard check: no invalid nodes inside d
V_set = set(V)
bad_arcs = [(u, v) for (u, v) in d.keys() if (u not in V_set or v not in V_set)]
if bad_arcs:
    raise ValueError(f"found {len(bad_arcs)} arcs with invalid nodes. example: {bad_arcs[:5]}")

# ------------------------------------------------------------
# service time S(i, j, t) (hours)
# only for (route, depot) pairs that survived filtering
# ------------------------------------------------------------

service_time = {}

required_cols = [
    "t_total_min_Car",
    "t_total_min_Moped",
    "t_total_min_EBikeCart",
    "t_total_min_FootBike",
]

missing_cols = [c for c in required_cols if c not in df_dr.columns]
if missing_cols:
    raise ValueError(f"missing required columns in delivery_times_2028.xlsx: {missing_cols}")

for _, r in df_dr.iterrows():
    i, j = r["route_clean"], r["depot_clean"]

    times_min = {
        "Car":       to_float(r["t_total_min_Car"]),
        "Moped":     to_float(r["t_total_min_Moped"]),
        "EBikeCart": to_float(r["t_total_min_EBikeCart"]),
        "FootBike":  to_float(r["t_total_min_FootBike"]),
    }

    missing_t = [t for t, v in times_min.items() if v is None]
    if missing_t:
        raise ValueError(f"missing service time(s) for ({i},{j}): {missing_t}")

    for t, minutes in times_min.items():
        service_time[(i, j, t)] = minutes / 60.0

# ------------------------------------------------------------
# demand D_i
# only for routes in I
# ------------------------------------------------------------

D = {}
for _, r in df_dem.iterrows():
    rid = r["route_clean"]
    bags = r["DeliveryBags"]
    if pd.isna(bags):
        raise ValueError(f"missing DeliveryBags for route {rid}")
    D[rid] = int(bags)

# sanity checks
if set(D.keys()) != set(I):
    missing_in_D = [i for i in I if i not in D]
    extra_in_D = [k for k in D.keys() if k not in set(I)]
    raise ValueError(f"demand mismatch. missing_in_D={missing_in_D[:10]}, extra_in_D={extra_in_D[:10]}")

if set(xy.keys()) != set(I):
    missing_in_xy = [i for i in I if i not in xy]
    extra_in_xy = [k for k in xy.keys() if k not in set(I)]
    raise ValueError(f"xy mismatch. missing_in_xy={missing_in_xy[:10]}, extra_in_xy={extra_in_xy[:10]}")

print(f"[ok] |d|={len(d)} distance arcs")
print(f"[ok] |service_time|={len(service_time)} entries")
print(f"[ok] |D|={len(D)} demand entries")
print(f"[ok] |xy|={len(xy)} xy entries")

# ------------------------------------------------------------
# bundle for pickle
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
        "service_time": service_time,
        "D": D,
        "xy": xy,
    }
}

# ------------------------------------------------------------
# save pickle
# ------------------------------------------------------------

with open(OUT_FILE, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n[ok] pickle written to {OUT_FILE}")
print(f"[final] |I|={len(I)}, |J|={len(J)}, |V|={len(V)}, |arcs|={len(d)}, |service_time|={len(service_time)}")
