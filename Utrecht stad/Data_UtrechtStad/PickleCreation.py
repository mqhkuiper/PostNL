# ============================================================
# Data loading + arc construction -> pickle  (FIXED)
# ============================================================

import pandas as pd
import pickle
from pathlib import Path
# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent

# Excel file in the same folder
PATH_DEPOT_ROUTE =  BASE_DIR / "delivery_times_2028.xlsx"
PATH_ROUTE_ROUTE =  BASE_DIR / "OD_RouteRoute.xlsx"
PATH_DEMAND      =  BASE_DIR / "Demand2028.xlsx"

OUT_FILE =  BASE_DIR / "model_input_US.pkl"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def to_float(x):
    """Conversion for Excel values that use comma decimals."""
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip()
        # Handles "9.726,46" or "4,999.93" etc.
        x = x.replace(".", "").replace(",", ".") if "," in x else x
    try:
        return float(x)
    except Exception:
        return None

def clean_id(x) -> str:
    """Normalize Excel IDs: 256, '256', 256.0 -> '256' ; strip whitespace."""
    if pd.isna(x):
        return None
    # Convert floats like 256.0 to int
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    s = str(x).strip()
    # Convert '256.0' -> '256'
    if s.endswith(".0"):
        s = s[:-2]
    return s

def route_id(x) -> str:
    s = clean_id(x)
    if s is None:
        return None
    return s

def depot_id(x) -> str:
    s = clean_id(x)
    if s is None:
        return None
    return f"D{s}"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df_dr  = pd.read_excel(PATH_DEPOT_ROUTE)
df_rr  = pd.read_excel(PATH_ROUTE_ROUTE)
df_dem = pd.read_excel(PATH_DEMAND)

# ------------------------------------------------------------
# Clean/standardize key columns (important!)
# ------------------------------------------------------------
df_dr["route_id_clean"] = df_dr["route_id"].apply(route_id)
df_dr["depot_id_clean"] = df_dr["depot_id"].apply(depot_id)

df_rr["from_clean"] = df_rr["FromRoute"].apply(route_id)
df_rr["to_clean"]   = df_rr["ToRoute"].apply(route_id)

df_dem["route_clean"] = df_dem["route_i"].apply(route_id)

# Drop rows with missing keys (if any)
df_dr = df_dr.dropna(subset=["route_id_clean", "depot_id_clean"]).copy()
df_rr = df_rr.dropna(subset=["from_clean", "to_clean"]).copy()
df_dem = df_dem.dropna(subset=["route_clean"]).copy()

# ------------------------------------------------------------
# Sets
# ------------------------------------------------------------
I = sorted(df_dem["route_clean"].unique())
J = sorted(set(df_dr["depot_id_clean"].unique()))
V = sorted(set(I) | set(J))

T = ["Car", "Moped", "EBikeCart", "FootBike"]

# ------------------------------------------------------------
# Unique pair counts (for reconciliation)
# ------------------------------------------------------------
pairs_dr = list(zip(df_dr["route_id_clean"], df_dr["depot_id_clean"]))
pairs_rr = list(zip(df_rr["from_clean"], df_rr["to_clean"]))

unique_dr = set(pairs_dr)
unique_rr = set(pairs_rr)

print("\n=== INPUT ROW COUNTS (raw) ===")
print(f"delivery_times rows (route-depot): {len(df_dr)}")
print(f"OD_RouteRoute rows (route-route) : {len(df_rr)}")

print("\n=== UNIQUE PAIR COUNTS (after ID cleaning) ===")
print(f"unique route-depot pairs: {len(unique_dr)}")
print(f"unique route-route pairs: {len(unique_rr)}")
print(f"duplicates route-depot: {len(df_dr) - len(unique_dr)}")
print(f"duplicates route-route: {len(df_rr) - len(unique_rr)}")

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

# ---- Distances d_ij (km)
d = {}

# Route–Route
for _, r in df_rr.iterrows():
    i = r["from_clean"]
    j = r["to_clean"]
    dist_m = to_float(r["Distance"])
    if dist_m is None:
        raise ValueError(f"Missing/invalid route-route distance for ({i},{j})")
    dist_km = dist_m / 1000.0

    # Protect against accidental overwrites with different values
    key = (i, j)
    if key in d and abs(d[key] - dist_km) > 1e-9:
        raise ValueError(f"Conflicting distances for arc {key}: {d[key]} vs {dist_km}")
    d[key] = dist_km

# Route–Depot (bidirectional distance)
for _, r in df_dr.iterrows():
    i = r["route_id_clean"]
    j = r["depot_id_clean"]
    dist_m = to_float(r["distance_m"])
    if dist_m is None:
        raise ValueError(f"Missing/invalid route-depot distance for ({i},{j})")
    dist_km = dist_m / 1000.0

    for key in [(i, j), (j, i)]:
        if key in d and abs(d[key] - dist_km) > 1e-9:
            raise ValueError(f"Conflicting distances for arc {key}: {d[key]} vs {dist_km}")
        d[key] = dist_km

# ---- Service time service_time_(route, depot, t) (hours)
# Interpretation: time to deliver all post of route i when supplied from depot j using mode t
service_time = {}

for _, r in df_dr.iterrows():
    i = r["route_id_clean"]
    j = r["depot_id_clean"]

    times_min = {
        "Car":       to_float(r.get("t_total_min_Car")),
        "Moped":     to_float(r.get("t_total_min_Moped")),
        "EBikeCart": to_float(r.get("t_total_min_EBikeCart")),
        "FootBike":  to_float(r.get("t_total_min_FootBike")),
    }

    # Enforce completeness so counts can match exactly
    missing = [t for t, v in times_min.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing service time(s) for route-depot pair ({i},{j}): {missing}"
        )

    for t, minutes in times_min.items():
        service_time[(i, j, t)] = minutes / 60.0   # one-way only

# ---- Demand D_i (delivery bags)
D = {}
for _, r in df_dem.iterrows():
    rid = r["route_clean"]
    bags = r["DeliveryBags"]
    if pd.isna(bags):
        raise ValueError(f"Missing DeliveryBags for route {rid}")
    D[rid] = int(bags)

# ------------------------------------------------------------
# Reconciliation checks (your expected formulas)
# ------------------------------------------------------------
expected_arcs = 2 * len(unique_dr) + len(unique_rr)
expected_service = len(unique_dr) * len(T)

print("\n=== EXPECTED COUNTS (from unique pairs) ===")
print(f"Expected arcs |d| = 2*|route-depot| + |route-route| = {expected_arcs}")
print(f"Expected service_time |S| = |route-depot|*|T| = {expected_service}")

print("\n=== ACTUAL COUNTS (built dicts) ===")
print(f"Actual arcs |d| = {len(d)}")
print(f"Actual service_time |S| = {len(service_time)}")
print(f"Actual demand |D| = {len(D)}")

if len(d) != expected_arcs:
    raise ValueError(f"Arc count mismatch: expected {expected_arcs}, got {len(d)}")

if len(service_time) != expected_service:
    raise ValueError(f"Service time count mismatch: expected {expected_service}, got {len(service_time)}")

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
        "service_time": service_time,  # renamed
        "D": D,
    }
}

# ------------------------------------------------------------
# Save pickle
# ------------------------------------------------------------
with open(OUT_FILE, "wb") as f:
    pickle.dump(model_data, f)

print(f"\nPickle written to {OUT_FILE}")
print(f"|I|={len(I)}, |J|={len(J)}, |V|={len(V)}, |arcs|={len(d)}, |service_time|={len(service_time)}")