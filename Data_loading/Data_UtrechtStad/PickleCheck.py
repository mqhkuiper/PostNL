# ============================================================
# Overview / sanity-check script for model_inputBGU.pkl
# ============================================================

import pickle
from itertools import islice
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent

PICKLE_NAME = "model_inputUS.pkl"

# ------------------------------------------------------------
# Load pickle
# ------------------------------------------------------------

with open(BASE_DIR / PICKLE_NAME, "rb") as f:
    data = pickle.load(f)

print("\n=== CONTENTS OF PICKLE ===")
for k, v in data.items():
    print(f"{k}: {type(v)}")

# ------------------------------------------------------------
# Unpack
# ------------------------------------------------------------

sets = data["sets"]
params = data["parameters"]

I  = sets["I"]                 # routes
J  = sets["J"]                 # depots
V  = sets["V"]                 # nodes
T  = sets["T"]                 # vehicle modes

d  = params["d"]               # distances (km)
S  = params["service_time"]    # service time (hours)
D  = params["D"]               # demand (delivery bags)
xy = params.get("xy", {})      # X/Y assignment (NEW)

A = list(d.keys())             # arc set derived from distances

# ------------------------------------------------------------
# Set sizes
# ------------------------------------------------------------

print("\n--- SET SIZES ---")
print(f"|I| (routes)      : {len(I)}")
print(f"|J| (depots)      : {len(J)}")
print(f"|V| (nodes)       : {len(V)}")
print(f"|A| (arcs)        : {len(A)}")
print(f"|T| (modes)       : {len(T)}  -> {T}")

# ------------------------------------------------------------
# Parameter sizes
# ------------------------------------------------------------

print("\n--- PARAMETER SIZES ---")
print(f"|D|  (demand entries)        : {len(D)}")
print(f"|d|  (distance arcs)         : {len(d)}")
print(f"|S|  (service-time entries)  : {len(S)}")
print(f"|xy| (XY assignments)        : {len(xy)}")

# ------------------------------------------------------------
# XY checks (CRUCIAL)
# ------------------------------------------------------------

print("\n--- XY ROUTE CHECKS ---")

# every route must have XY code
missing_xy = [i for i in I if i not in xy]
if not missing_xy:
    print("OK: all routes in I have an XY code")
else:
    print("ERROR: routes without XY code found!")
    print("Example:", missing_xy[:5])

# no extra XY routes outside I
extra_xy = [i for i in xy if i not in I]
if not extra_xy:
    print("OK: no XY routes outside route set I")
else:
    print("WARNING: XY contains routes not in I")
    print("Example:", extra_xy[:5])

# distribution
xy_count = Counter(xy[i] for i in I)
print(f"XY distribution: {dict(xy_count)}")

# ------------------------------------------------------------
# Sample XY assignments
# ------------------------------------------------------------

print("\n--- SAMPLE XY ASSIGNMENTS (route -> X/Y) ---")
for item in islice(xy.items(), 10):
    print(item)
# ------------------------------------------------------------
# Sample elements
# ------------------------------------------------------------

print("\n--- SAMPLE ELEMENTS ---")
print("Routes (I):", list(islice(I, 5)))
print("Depots (J):", list(islice(J, 5)))
print("Nodes (V):", list(islice(V, 5)))
print("Arcs (A):", list(islice(A, 5)))

# ------------------------------------------------------------
# Demand samples
# ------------------------------------------------------------

print("\n--- SAMPLE DEMAND D_i (delivery bags) ---")
for item in islice(D.items(), 5):
    print(item)

# ------------------------------------------------------------
# Distance samples
# ------------------------------------------------------------

print("\n--- SAMPLE DISTANCES d_ij (km) ---")
for item in islice(d.items(), 5):
    print(item)

# ------------------------------------------------------------
# Service-time samples
# ------------------------------------------------------------

print("\n--- SAMPLE SERVICE TIMES S_(route,depot,mode) (hours) ---")
for item in islice(S.items(), 5):
    print(item)

# ------------------------------------------------------------
# Structural sanity checks
# ------------------------------------------------------------

print("\n--- STRUCTURAL CHECKS ---")

# 1) All service times must be route -> depot
bad_keys = [k for k in islice(S.keys(), 20000) if k[0].startswith("D")]
if not bad_keys:
    print("OK: service_time keys are route -> depot only")
else:
    print("ERROR: Found depot -> route service_time keys!")
    print("Example:", bad_keys[0])

# 2) Demand keys must match route set
bad_demand = [i for i in D if i not in I]
if not bad_demand:
    print("OK: demand defined only for routes in I")
else:
    print("ERROR: demand found for routes not in I")
    print("Example:", bad_demand[:5])

# 3) Arc endpoint check
bad_arcs = [(u, v) for (u, v) in islice(A, 50000) if u not in V or v not in V]
if not bad_arcs:
    print("OK: all arcs connect valid nodes")
else:
    print("ERROR: arcs with invalid nodes found")
    print("Example:", bad_arcs[0])

# ------------------------------------------------------------
# Arc logic reminder
# ------------------------------------------------------------

print("\nArc logic reminder:")
print("  |d| = 2 × (#unique route–depot pairs) + (#route–route pairs)")
print("  |S| = (#unique route–depot pairs) × |T|")

print("\n=== PICKLE CHECK COMPLETE — STRUCTURE CONSISTENT ===")
