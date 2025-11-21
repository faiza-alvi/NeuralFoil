import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

# ----------------------------------------------------
# Load CSV with Polars and select only first 18 columns
# ----------------------------------------------------
df = pl.read_csv("training/training_data/data_xfoil_clean.csv", has_header=False, infer_schema_length=10000, skip_rows=1)

# Convert just the first 18 columns to a numpy array (float)
params_all = df.select(df.columns[:18]).to_numpy()

# ----------------------------------------------------
# Extract unique rows
# ----------------------------------------------------
unique_params = np.unique(params_all, axis=0)

print(f"Found {len(unique_params)} unique Kulfan parameter sets.")

# ----------------------------------------------------
# Plotting
# ----------------------------------------------------
plt.figure(figsize=(8, 4))

for params in unique_params:
    ku_upper = params[0:8]
    ku_lower = params[8:16]
    t_le     = params[16]
    t_te     = params[17]

    # Generate Kulfan coordinates (Nx2 array)
    coords = get_kulfan_coordinates(
        upper_weights=ku_upper,
        lower_weights=ku_lower,
        leading_edge_weight=t_le,
        TE_thickness=t_te,
        n_points_per_side=200
    )

    x = coords[:, 0]
    y = coords[:, 1]

    plt.plot(x, y, color="blue", alpha=0.15, linewidth=1)


# ----------------------------------------------------
# Formatting
# ----------------------------------------------------
plt.gca().set_aspect("equal", adjustable="box")
plt.title("Kulfan Airfoils (from Polars CSV)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
