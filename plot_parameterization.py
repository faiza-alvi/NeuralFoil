import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd
import csv
from aerosandbox.geometry.airfoil.airfoil_families import (
    get_kulfan_parameters,
    get_kulfan_coordinates,
)

airfoil_path = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\BirdAirfoils\Tyto_alba_NC17725_0.50.csv"
with open(airfoil_path, mode='r') as f:
    csv_reader = csv.reader(f)
    data = [row for row in csv_reader]

df_array = np.array(data).astype(float)
df_array = df_array[::-1]

af = asb.Airfoil(coordinates=df_array)

N = 8

kulfan_parameters = get_kulfan_parameters(af.coordinates, n_weights_per_side=N,)
lines = []


# ============================================================
# Original airfoil
# ============================================================

coords = np.asarray(af.coordinates)

for point, (x, y) in enumerate(coords):
    lines.append({
        "mode": "original",
        "parameter": "original",
        "point": point,
        "x": x,
        "y": y
    })


# ============================================================
# Upper-surface Kulfan modes
# ============================================================

for i in range(N):

    # This matches the indexing in your original code
    parameter_name = f"upper_{N - i}"

    coords = np.asarray(
        get_kulfan_coordinates(
            lower_weights=kulfan_parameters["lower_weights"],
            upper_weights=(
                kulfan_parameters["upper_weights"]
                + np.eye(N)[N - i - 1] * 1
            ),
            leading_edge_weight=kulfan_parameters["leading_edge_weight"],
            TE_thickness=kulfan_parameters["TE_thickness"],
        )
    )

    for point, (x, y) in enumerate(coords):
        lines.append({
            "mode": "upper",
            "parameter": parameter_name,
            "point": point,
            "x": x,
            "y": y
        })


# ============================================================
# Lower-surface Kulfan modes
# ============================================================

for i in range(N):

    parameter_name = f"lower_{i + 1}"

    coords = np.asarray(
        get_kulfan_coordinates(
            lower_weights=(
                kulfan_parameters["lower_weights"]
                + np.eye(N)[i] * -1
            ),
            upper_weights=kulfan_parameters["upper_weights"],
            leading_edge_weight=kulfan_parameters["leading_edge_weight"],
            TE_thickness=kulfan_parameters["TE_thickness"],
        )
    )

    for point, (x, y) in enumerate(coords):
        lines.append({
            "mode": "lower",
            "parameter": parameter_name,
            "point": point,
            "x": x,
            "y": y
        })


# ============================================================
# Leading-edge modification
# ============================================================

coords = np.asarray(
    get_kulfan_coordinates(
        lower_weights=kulfan_parameters["lower_weights"],
        upper_weights=kulfan_parameters["upper_weights"],
        leading_edge_weight=(
            kulfan_parameters["leading_edge_weight"] + 1
        ),
        TE_thickness=kulfan_parameters["TE_thickness"],
    )
)

for point, (x, y) in enumerate(coords):
    lines.append({
        "mode": "leading_edge",
        "parameter": "leading_edge",
        "point": point,
        "x": x,
        "y": y
    })


# ============================================================
# Trailing-edge modification
# ============================================================

coords = np.asarray(
    get_kulfan_coordinates(
        lower_weights=kulfan_parameters["lower_weights"],
        upper_weights=kulfan_parameters["upper_weights"],
        leading_edge_weight=kulfan_parameters["leading_edge_weight"],
        TE_thickness=(
            kulfan_parameters["TE_thickness"] + 0.2
        ),
    )
)

for point, (x, y) in enumerate(coords):
    lines.append({
        "mode": "trailing_edge",
        "parameter": "trailing_edge",
        "point": point,
        "x": x,
        "y": y
    })


# ============================================================
# Create dataframe and save
# ============================================================

df = pd.DataFrame(lines)

# ============================================================
# Quick plot check
# ============================================================

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colors = ["orange", "darkseagreen", "dodgerblue"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

fig, ax = plt.subplots(figsize=(6, 4.5))

# Original airfoil
original = df[df["mode"] == "original"]

ax.plot(
    original["x"],
    original["y"],
    "-k",
    linewidth=1.5,
    zorder=10
)

ax.fill(
    original["x"],
    original["y"],
    color="black",
    alpha=0.15
)


# Upper modes
upper = df[df["mode"] == "upper"]

for i, parameter in enumerate(
    sorted(upper["parameter"].unique(), reverse=True)
):
    group = upper[upper["parameter"] == parameter]

    ax.plot(
        group["x"],
        group["y"],
        color=cmap(i / (N - 1)),
        alpha=0.7,
        linewidth=1
    )


# Lower modes
lower = df[df["mode"] == "lower"]

for i, parameter in enumerate(
    sorted(lower["parameter"].unique())
):
    group = lower[lower["parameter"] == parameter]

    ax.plot(
        group["x"],
        group["y"],
        color=cmap((i + N) / (2 * N)),
        alpha=0.7,
        linewidth=1
    )


# Leading edge
lem = df[df["mode"] == "leading_edge"]

ax.plot(
    lem["x"],
    lem["y"],
    color="red",#cmap(1.0),
    alpha=0.7,
    linewidth=1
)


# Trailing edge
te = df[df["mode"] == "trailing_edge"]

ax.plot(
    te["x"],
    te["y"],
    color="purple", #cmap(0.0),
    alpha=0.7,
    linewidth=1
)


ax.set_aspect("equal")
ax.set_xlim(-0.05, 1.05)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Kulfan Parameterization Check")

plt.tight_layout()
plt.show()


# ============================================================
# Save CSV
# ============================================================
output_path = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\BirdData"

df.to_csv(
    "Tyto_alba_NC17725_0.50_kulfan_param_lines.csv",
    index=False
)

print("Saved:", "Tyto_alba_NC17725_0.50_kulfan_param_lines.csv")