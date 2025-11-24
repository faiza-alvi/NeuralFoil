import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
from neuralfoil._basic_data_type import Data
from pathlib import Path
import aerosandbox as asb
import os

# ----------------------------------------------------
# Load CSV with Polars and select only first 18 columns
# ----------------------------------------------------
cols = Data.get_vector_column_names()

### Read the original data, by scraping all .csv files within the data directory
# data_directory = Path(r"/home/faiza/Documents/NeuralFoil/training/training_data")
# data_directory = Path(r"/home/faiza/Downloads/training_data/training_data")

# airfoil_database_path = Path("/home/faiza/Documents/TrainingAirfoils")
data_directory = Path("/home/faiza/Documents/NeuralFoil")

# ________________________________________
# USE BELOW WHEN CYCLING THROUGH .DAT FILES OF AIRFOILS 

# def load_airfoil_coordinates(filepath):
#     # Skip the first line (airfoil name), and load the x,y columns
#     with open(filepath, 'r') as f:
#         lines = f.readlines()

#     # Remove any empty lines and non-numeric lines
#     coords = []
#     for line in lines[1:]:  # skip the first line (usually name/title)
#         try:
#             x, y = map(float, line.strip().split())
#             coords.append([x, y])
#         except ValueError:
#             continue  # Skip lines that can't be parsed

#     return np.array(coords)

# airfoil_database = [
#     asb.Airfoil(
#         name=filename.stem,
#         coordinates=load_airfoil_coordinates(filename)
#     ).normalize().to_kulfan_airfoil()
#     for filename in airfoil_database_path.glob("*.dat")
# ]

# ### Compute the covariance matrix of airfoil shape parameters, for better data generation later
# kulfans_database = np.stack(
#     [
#         np.concatenate(
#             [
#                 airfoil.upper_weights,
#                 airfoil.lower_weights,
#                 np.atleast_1d(airfoil.leading_edge_weight),
#                 np.atleast_1d(airfoil.TE_thickness),
#             ]
#         )
#         for airfoil in airfoil_database
#     ],
#     axis=0,
# )
# mean_database = np.mean(kulfans_database, axis=0)
# cov_database = np.cov(kulfans_database, rowvar=False)

# print("mean and cov of kulfans database")
# print(mean_database)
# print(cov_database)

# USE BELOW WHEN WORKING WITH .CSV 
raw_dfs = {}

for csv_file in data_directory.glob("data*.csv"):
    print(f"Reading {csv_file}...")
    raw_dfs[csv_file.stem] = pl.read_csv(
        csv_file, has_header=False, new_columns=cols, schema_overrides={col: pl.Float32 for col in cols}, infer_schema_length=10000, skip_rows=1
    )
    print(f"\t{len(raw_dfs[csv_file.stem])} rows")

df = pl.concat(raw_dfs.values())

# Do some basic cleanup
cols_to_nullify = Data.get_vector_output_column_names().copy()
cols_to_nullify.remove("analysis_confidence")

c = pl.col("CD") <= 0
print(f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with CD <= 0...")
df = df.with_columns(
    [
        pl.when(c)
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal(
    [pl.col(f"upper_bl_theta_{i}") <= 0 for i in range(Data.N)]
    + [pl.col(f"lower_bl_theta_{i}") <= 0 for i in range(Data.N)]
)
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with nonpositive boundary layer thetas..."
)
df = df.with_columns(
    [
        pl.when(c)
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal(
    [pl.col(f"upper_bl_H_{i}") < 1 for i in range(Data.N)]
    + [pl.col(f"lower_bl_H_{i}") < 1 for i in range(Data.N)]
)
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with H < 1 (non-physical BL)..."
)
df = df.with_columns(
    [
        pl.when(c)
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal(
    sum(
        [
            [
                pl.col(f"upper_bl_ue/vinf_{i}") < -20,
                pl.col(f"upper_bl_ue/vinf_{i}") > 20,
                pl.col(f"lower_bl_ue/vinf_{i}") < -20,
                pl.col(f"lower_bl_ue/vinf_{i}") > 20,
            ]
            for i in range(Data.N)
        ],
        start=[],
    )
)
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with non-physical edge velocities..."
)
df = df.with_columns(
    [
        pl.when(c)
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

c = pl.any_horizontal(
    pl.col("Top_Xtr") < 0,
    pl.col("Top_Xtr") > 1,
    pl.col("Bot_Xtr") < 0,
    pl.col("Bot_Xtr") > 1,
)
print(
    f"Nullifying {int(df.select(c).sum().to_numpy()[0, 0])} rows with non-physical transition locations..."
)
df = df.with_columns(
    [
        pl.when(c)
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(c).then(None).otherwise(pl.col(col)).alias(col)
        for col in cols_to_nullify
    ]
)

# Convert just the first 18 columns to a numpy array (float)
params_all = df.select(df.columns[:18]).to_numpy()

# ----------------------------------------------------
# Extract unique rows
# ----------------------------------------------------
unique_params = np.unique(params_all, axis=0)

# unique_params = kulfans_database

zscores = (unique_params - unique_params.mean(axis=0)) / unique_params.std(axis=0)
outlier_mask = np.any(np.abs(zscores) > 3 , axis=1)
#outlier_mask = np.any((np.abs(zscores) > 5) & (np.abs(zscores) < 10), axis=1)


print("Outlier Rows")
print(unique_params[outlier_mask])
idx = np.where(outlier_mask)[0]
print(idx)
print(len(idx))

# for i in idx:
#     print(i, airfoil_database[i].name)

print(f"Found {len(unique_params)} unique Kulfan parameter sets.")

# ----------------------------------------------------
# Plotting only NON-outlier airfoils
# ----------------------------------------------------
# plt.figure(figsize=(8, 4))

clean_params = unique_params[~outlier_mask]   # <--- THIS IS THE KEY LINE
print("The number of non outliers are: ", len(clean_params))


# for params in kulfans_database:
#     ku_upper = params[0:8]
#     ku_lower = params[8:16]
#     t_le     = params[16]
#     t_te     = params[17]

#     coords = get_kulfan_coordinates(
#         upper_weights=ku_upper,
#         lower_weights=ku_lower,
#         leading_edge_weight=t_le,
#         TE_thickness=t_te,
#         n_points_per_side=200
#     )

#     x = coords[:, 0]
#     y = coords[:, 1]

#     plt.plot(x, y, color="blue", alpha=0.15, linewidth=1)
#     plt.ylim((-1,1))
#     plt.gca().set_aspect("equal", adjustable="box")
        

# plt.show()

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

plt.gca().set_aspect("equal", adjustable="box")
plt.title("Kulfan Airfoils in Training Data")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------
# New section that plots and saves all airfoils in the database
# print("len(airfoil_database) =", len(airfoil_database))
# print("len(kulfans_database) =", len(kulfans_database))


# save_dir = "/home/faiza/Documents/AirfoilDatabasePlots"
# os.makedirs(save_dir, exist_ok=True)   # Create folder if missing

# for i, params in enumerate(kulfans_database):

#     ku_upper = params[0:8]
#     ku_lower = params[8:16]
#     t_le     = params[16]
#     t_te     = params[17]

#     coords = get_kulfan_coordinates(
#         upper_weights=ku_upper,
#         lower_weights=ku_lower,
#         leading_edge_weight=t_le,
#         TE_thickness=t_te,
#         n_points_per_side=200
#     )

#     x = coords[:, 0]
#     y = coords[:, 1]

#     # New figure for each airfoil
#     plt.figure(figsize=(6, 3))
#     plt.plot(x, y, linewidth=1)
#     plt.ylim((-1, 1))
#     plt.gca().set_aspect("equal", adjustable="box")

#     name = airfoil_database[i].name
#     plt.title(name)

#     # Full file path
#     filepath = os.path.join(save_dir, f"{name}.png")

#     plt.savefig(filepath, dpi=300, bbox_inches="tight")
#     plt.close()

# print(i, "finished")
