import pickle
from itertools import islice

# ------------------------------------------------------------
# Load pickle
# ------------------------------------------------------------
with open("model_input.pkl", "rb") as f:
    data = pickle.load(f)

print("\n=== CONTENTS OF model_input.pkl ===")
for k, v in data.items():
    print(f"{k}: {type(v)}")

# ------------------------------------------------------------
# Unpack
# ------------------------------------------------------------
sets = data["sets"]
params = data["parameters"]

I = sets["I"]
J = sets["J"]
V = sets["V"]
T = sets["T"]

d = params["d"]     # distances
L = params["L"]     # lead times
D = params["D"]     # demand

A = list(d.keys())  # arc set derived from distance dictionary

# ------------------------------------------------------------
# Set sizes
# ------------------------------------------------------------
print("\n--- SET SIZES ---")
print(f"|I| (routes)  : {len(I)}")
print(f"|J| (depots)  : {len(J)}")
print(f"|V| (nodes)   : {len(V)}")
print(f"|A| (arcs)    : {len(A)}")
print(f"T (modes)    : {T}")

# ------------------------------------------------------------
# Sample elements
# ------------------------------------------------------------
print("\n--- SAMPLE ELEMENTS ---")
print("Routes (I):", list(islice(I, 5)))
print("Depots (J):", list(islice(J, 5)))
print("Nodes (V):", list(islice(V, 5)))
print("Arcs (A):", list(islice(A, 5)))

# ------------------------------------------------------------
# Demand
# ------------------------------------------------------------
print("\n--- DEMAND D_i (delivery bags) ---")
print(f"Number of demand entries |D| = {len(D)}")
for item in islice(D.items(), 5):
    print(item)

# ------------------------------------------------------------
# Distances
# ------------------------------------------------------------
print("\n--- DISTANCES d_ij (km) ---")
print(f"Number of distance arcs |d| = {len(d)}")
for item in islice(d.items(), 5):
    print(item)

# ------------------------------------------------------------
# Travel / lead times
# ------------------------------------------------------------
print("\n--- LEAD TIMES L_ijt (hours) ---")
print(f"Number of lead time entries |L| = {len(L)}")
for item in islice(L.items(), 5):
    print(item)
