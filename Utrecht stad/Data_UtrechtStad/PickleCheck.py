# ============================================================
# Overview / sanity-check script for model_input.pkl
# ============================================================

import pickle
from itertools import islice
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# ------------------------------------------------------------
# Load pickle
# ------------------------------------------------------------
with open(BASE_DIR/ "model_input_US.pkl", "rb") as f:
    data = pickle.load(f)

print("\n=== CONTENTS OF model_input.pkl ===")
for k, v in data.items():
    print(f"{k}: {type(v)}")

# ------------------------------------------------------------
# Unpack
# ------------------------------------------------------------
sets = data["sets"]
params = data["parameters"]

I = sets["I"]                 # routes
J = sets["J"]                 # depots
V = sets["V"]                 # nodes
T = sets["T"]                 # vehicle modes

d = params["d"]               # distances (km)
S = params["service_time"]    # service time (hours)
D = params["D"]               # demand (delivery bags)

A = list(d.keys())            # arc set derived from distances

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

# 1) All service times must be route -> depot (never depot -> route)
bad_keys = [k for k in islice(S.keys(), 20000) if k[0].startswith("D")]
if len(bad_keys) == 0:
    print("OK: service_time keys are route -> depot only")
else:
    print("ERROR: Found depot -> route service_time keys!")
    print("Example:", bad_keys[0])

# 2) Arc logic explanation
print("\nArc logic reminder:")
print("  |d| = 2 × (#unique route–depot pairs) + (#route–route pairs)")
print("  |S| = (#unique route–depot pairs) × |T|")

print("\n=== OVERVIEW COMPLETE ===")
