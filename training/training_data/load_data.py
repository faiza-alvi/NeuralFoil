import aerosandbox.numpy as np
import polars as pl
import pandas as pd 
from scipy.special import comb
from pathlib import Path
from neuralfoil._basic_data_type import Data
import torch
import math
import time
import shutil
import gc

start_time = time.time() 

# Function that calculates the derivatives of the airfoil using exact differentiation at specific nodes. 
# With 8 polynomials per side, the first and second derivatives are calculated at 1/7, 2/7, 3/7, 4/7, 5/7 
# and 6/7 along the chord
def derivatives_at_nodes(lower_weights: np.ndarray = -0.2 * np.ones(8),
    upper_weights: np.ndarray = 0.2 * np.ones(8),
    leading_edge_weight: float = 0.0,
    TE_thickness: float = 0.0,
    **deprecated_kwargs,
) -> np.ndarray:
    """
    Given a set of Kulfan Parameters (18 values)
    Finds the first and second derivative of the surface at specific nodes along the airfoil
    """
    N1 = 0.5 
    N2 = 1.0,

    n_weights_per_side = len(lower_weights)

    x_nodes = np.linspace(0, 1, n_weights_per_side)
    
    x_nodes = x_nodes[1:-1] #eliminates end points and only uses interior nodes for the derivative calculation. 

    n_nodes = len(x_nodes)

    N = n_weights_per_side - 1  # Order of Bernstein polynomials

    K = comb(N, np.arange(N + 1))  # Bernstein polynomial coefficients

    dims = (n_weights_per_side, n_nodes)

    def wide(vector):
        return np.tile(np.reshape(vector, (1, dims[1])), (dims[0], 1))

    def tall(vector):
        return np.tile(np.reshape(vector, (dims[0], 1)), (1, dims[1]))

    p = np.arange(N1, (N1 + N + 1), 1) #exponent of x
    q = N-np.arange(N + 1) + N2 #exponent of 1-x
    
    p_1 = p - 1 
    q_1 = q - 1 
    p_2 = p - 2
    q_2 = q - 2
    q_2[q_2 < 0] = 0 #hardcoding that q-2 after zero goes to zero still since derivative of a constant is zero. 
    
    slopes_matrix = (
        tall(K)
        * tall(p) 
        * wide(x_nodes) ** tall(p_1) 
        * wide( 1-x_nodes ) ** tall(q) 
        - 
        tall(K)
        * tall(q)
        * wide(x_nodes) ** tall(p) 
        * wide( 1-x_nodes ) ** tall(q_1)
    )       

    curvature_matrix = (
        tall(K)
        * tall(p) * tall(p_1) 
        * wide(x_nodes) ** tall(p_2) 
        * wide( 1-x_nodes ) ** tall(q) 
        - 2 * tall(K)
        * tall(q)
        * tall(p)
        * wide(x_nodes) ** tall(p_1) 
        * wide( 1-x_nodes ) ** tall(q_1) 
        + tall(K) 
        * tall(q)
        * tall(q_1)
        * wide(x_nodes) ** tall(p)
        * wide( 1-x_nodes) ** tall(q_2)
    )

    lowerslope = slopes_matrix.T @ lower_weights
    lowercurve = curvature_matrix.T @ lower_weights
    upperslope = slopes_matrix.T @ upper_weights
    uppercurve = curvature_matrix.T @ upper_weights

    #Add in Leading Edge Modification
    m_upper = np.length(upper_weights) + 0.5
    m_lower = np.length(lower_weights) + 0.5
    LE_upper_slope = leading_edge_weight*((1 - x_nodes)**m_upper) - leading_edge_weight*m_upper*x_nodes*((1-x_nodes)**(m_upper-1))
    LE_lower_slope = leading_edge_weight*((1 - x_nodes)**m_lower) - leading_edge_weight*m_lower*x_nodes*((1-x_nodes)**(m_lower-1))
    LE_upper_curve = -2*leading_edge_weight*m_upper*((1-x_nodes)**(m_upper-1)) + leading_edge_weight*m_upper*(m_upper-1)*x_nodes*((1-x_nodes)**(m_upper-2))
    LE_lower_curve = -2*leading_edge_weight*m_lower*((1-x_nodes)**(m_lower-1)) + leading_edge_weight*m_lower*(m_lower-1)*x_nodes*((1-x_nodes)**(m_lower-2))
    upperslope = upperslope + LE_upper_slope
    lowerslope = lowerslope + LE_lower_slope
    uppercurve = uppercurve + LE_upper_curve
    lowercurve = lowercurve + LE_lower_curve

    #Add in Trailing Edge Thickness 
    upperslope = upperslope + TE_thickness/2
    lowerslope = lowerslope - TE_thickness/2
    # No effect on the second derivative as it is a function of x and so the second derivative is zero
    
    return {
        "upper_first_der": upperslope,
        "lower_first_der": lowerslope,
        "upper_second_der": uppercurve,
        "lower_second_der": lowercurve,
    }

# Given an input row of kulfan parameters from the dataframe this applies derivatives at nodes to create an output of derivative nodes. 
def compute_derivatives(row):
    result = derivatives_at_nodes(
        lower_weights=np.array([row[f"kulfan_lower_{i}"] for i in range(8)]),
        upper_weights=np.array([row[f"kulfan_upper_{i}"] for i in range(8)]),
        leading_edge_weight=row["kulfan_LE_weight"],
        TE_thickness=row["kulfan_TE_thickness"],
    )
    
    # Expand into flat dict with meaningful column names
    return {
        f"s_upper_first_der_{i}": result["upper_first_der"][i] for i in range(len(result["upper_first_der"]))
    } | {
        f"s_lower_first_der_{i}": result["lower_first_der"][i] for i in range(len(result["lower_first_der"]))
    } | {
        f"s_upper_second_der_{i}": result["upper_second_der"][i] for i in range(len(result["upper_second_der"]))
    } | {
        f"s_lower_second_der_{i}": result["lower_second_der"][i] for i in range(len(result["lower_second_der"]))
    }

# New method for derivative computation
# Uses GPU to accelerate calculation

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Precompute Bernstein Matrices ----------
def precompute_bernstein(n_weights=8):
    """
    Precompute slopes and curvature matrices for the Bernstein polynomials
    """
    N1 = 0.5
    N2 = 1.0

    x_nodes = torch.linspace(0, 1, n_weights, device=device)[1:-1]  # interior nodes
    N = n_weights - 1

    # Binomial coefficients
    K = torch.tensor([math.comb(N, i) for i in range(N+1)],
                     dtype=torch.float32, device=device)

    p = torch.arange(N1, N1+N+1, device=device, dtype=torch.float32)
    q = N - torch.arange(N+1, device=device, dtype=torch.float32) + N2

    p1 = p - 1
    q1 = q - 1
    p2 = p - 2
    q2 = q - 2
    q2[q2 < 0] = 0

    x = x_nodes.unsqueeze(0)  # shape (1, n_nodes)

    slopes = ((K[:,None]*p[:,None]*x**p1[:,None]*(1-x)**q[:,None])
              - (K[:,None]*q[:,None]*x**p[:,None]*(1-x)**q1[:,None])).T

    curvature = ((K[:,None]*p[:,None]*p1[:,None]*x**p2[:,None]*(1-x)**q[:,None])
                 - (2*K[:,None]*q[:,None]*p[:,None]*x**p1[:,None]*(1-x)**q1[:,None])
                 + (K[:,None]*q[:,None]*q1[:,None]*x**p[:,None]*(1-x)**q2[:,None])).T

    return slopes, curvature, x_nodes

# Precompute once
slopes_matrix, curvature_matrix, x_nodes = precompute_bernstein()

# ---------- GPU Batch Derivative Function ----------
def compute_derivatives_batch(lower, upper, LE, TE):
    """
    lower, upper: (batch_size, 8)
    LE, TE: (batch_size,)
    returns 4 tensors: upper_slope, lower_slope, upper_curve, lower_curve
    shape: (batch_size, n_nodes)
    """
    lowerslope = (slopes_matrix @ lower.T).T
    upperslope = (slopes_matrix @ upper.T).T
    lowercurve = (curvature_matrix @ lower.T).T
    uppercurve = (curvature_matrix @ upper.T).T

    m = lower.shape[1] + 0.5
    x = x_nodes

    LE = LE[:,None]

    LE_slope = LE*(1-x)**m - LE*m*x*(1-x)**(m-1)
    LE_curve = -2*LE*m*(1-x)**(m-1) + LE*m*(m-1)*x*(1-x)**(m-2)

    upperslope += LE_slope
    lowerslope += LE_slope
    uppercurve += LE_curve
    lowercurve += LE_curve

    upperslope += TE[:,None]/2
    lowerslope -= TE[:,None]/2

    return upperslope, lowerslope, uppercurve, lowercurve

# ---------- Batch Processing Function ----------
def process_airfoil_batches_to_disk(
    df: pl.DataFrame,
    output_dir: Path,
    batch_size: int = 50_000
):
    n = df.height
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = df[start:end]

        # Original row indices
        row_indices = np.arange(start, end)

        # Convert batch to GPU tensors
        lower = torch.tensor(
            batch.select([f"kulfan_lower_{i}" for i in range(8)]).to_numpy(),
            dtype=torch.float32, device=device
        )
        upper = torch.tensor(
            batch.select([f"kulfan_upper_{i}" for i in range(8)]).to_numpy(),
            dtype=torch.float32, device=device
        )
        LE = torch.tensor(batch["kulfan_LE_weight"].to_numpy(), dtype=torch.float32, device=device)
        TE = torch.tensor(batch["kulfan_TE_thickness"].to_numpy(), dtype=torch.float32, device=device)

        # Compute derivatives
        u_slope, l_slope, u_curve, l_curve = compute_derivatives_batch(lower, upper, LE, TE)
        n_nodes = u_slope.shape[1]

        # Save batch immediately
        derivatives_batch = pl.DataFrame({
            "row_index": row_indices,
            **{f"s_upper_first_der_{i}": u_slope[:, i].cpu().numpy() for i in range(n_nodes)},
            **{f"s_lower_first_der_{i}": l_slope[:, i].cpu().numpy() for i in range(n_nodes)},
            **{f"s_upper_second_der_{i}": u_curve[:, i].cpu().numpy() for i in range(n_nodes)},
            **{f"s_lower_second_der_{i}": l_curve[:, i].cpu().numpy() for i in range(n_nodes)},
        })
        derivatives_batch.write_parquet(output_dir / f"batch_{start:08d}_{end:08d}.parquet")
        print(f"Processed batch {start}-{end} / {n}")


cols = Data.get_vector_column_names()

### Read the original data, by scraping all .csv files within the data directory
# data_directory = Path(r"/home/faiza/Documents/NeuralFoil/training/training_data")
# data_directory = Path(r"/home/faiza/Downloads/training_data/training_data")
#data_directory = Path(r"/home/faiza/Documents/Gen2TrainingAirfoils")
data_directory = Path(r"/media/faiza/Seagate Portable Drive/Kaleb_documents/Gen2TrainingAirfoils")
# Folder to store derivative batches
derivatives_dir = Path("derivatives_batches")

# --- Step 1: Delete old derivatives if exist ---
if derivatives_dir.exists():
    print(f"Deleting old derivatives in {derivatives_dir}...")
    shutil.rmtree(derivatives_dir)
derivatives_dir.mkdir(exist_ok=True)

raw_dfs = {}

for csv_file in data_directory.glob("data*.csv"):
    print(f"Reading {csv_file}...")
    raw_dfs[csv_file.stem] = pl.read_csv(
        csv_file, has_header=False, new_columns=cols, dtypes={col: pl.Float32 for col in cols}, infer_schema_length=10000, skip_rows=1
    )
    print(f"\t{len(raw_dfs[csv_file.stem])} rows")

df = pl.concat(raw_dfs.values())

# Do some basic cleanup
cols_to_nullify = Data.get_vector_output_column_names().copy()
cols_to_nullify.remove("analysis_confidence")

# Set analysis confidence to zero for all rows with nonpositive drag coefficients 
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

# Set analysis confidence to 0 for all non positive boundary layer thetas 
df = df.lazy() #Makes the dataframe lazy to avoid rewriting too much 
# Compute the mask once and store it
df = df.with_columns(
    pl.any_horizontal(
        [pl.col(f"upper_bl_theta_{i}") <= 0 for i in range(Data.N)] +
        [pl.col(f"lower_bl_theta_{i}") <= 0 for i in range(Data.N)]
    ).alias("mask_c")
)

# Count rows to log
n_bad = df.select(pl.col("mask_c").sum()).collect().item()
print(f"Nullifying {n_bad} rows with nonpositive boundary layer thetas...")

# Apply transformations using the cached mask
df = df.with_columns(
    [
        pl.when(pl.col("mask_c"))
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(pl.col("mask_c"))
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
        for col in cols_to_nullify
    ]
)
# Drop the temporary mask column
df = df.drop("mask_c")
# Collect the lazy frame
df = df.collect()

# Set analysis confidence = 0 for all rows with non physical BL 
df = df.lazy()
df = df.with_columns(
    pl.any_horizontal(
        [pl.col(f"upper_bl_H_{i}") < 1 for i in range(Data.N)] +
        [pl.col(f"lower_bl_H_{i}") < 1 for i in range(Data.N)]
    ).alias("mask_H")
)
# Count affected rows without recomputing
n_bad_H = df.select(pl.col("mask_H").sum()).collect().item()
print(f"Nullifying {n_bad_H} rows with H < 1 (non-physical BL)...")
# Apply nullification / zeroing using the cached mask
df = df.with_columns(
    [
        pl.when(pl.col("mask_H"))
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(pl.col("mask_H"))
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
        for col in cols_to_nullify
    ]
)

# Drop the temporary mask column to save memory
df = df.drop("mask_H")
# Collect lazy frame (executes everything)
df = df.collect()

# Sets analysis confidence to 0 for all non-physical edge velocities 
# Make sure the DataFrame is lazy
df = df.lazy()
# Build mask_ue safely using vectorized expressions
mask_exprs = []
for i in range(Data.N):
    mask_exprs.extend([
        pl.col(f"upper_bl_ue/vinf_{i}") < -20,
        pl.col(f"upper_bl_ue/vinf_{i}") > 20,
        pl.col(f"lower_bl_ue/vinf_{i}") < -20,
        pl.col(f"lower_bl_ue/vinf_{i}") > 20,
    ])
df = df.with_columns(
    pl.any_horizontal(mask_exprs).alias("mask_ue")
)

# Count affected rows
n_bad_ue = df.select(pl.col("mask_ue").sum().alias("n")).collect()["n"][0]
print(f"Nullifying {n_bad_ue} rows with non-physical edge velocities...")

# Apply nullification / zeroing using the cached mask
df = df.with_columns(
    [
        pl.when(pl.col("mask_ue"))
        .then(0)
        .otherwise(pl.col("analysis_confidence"))
        .alias("analysis_confidence"),
    ]
    + [
        pl.when(pl.col("mask_ue"))
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
        for col in cols_to_nullify
    ]
)
# Drop the temporary mask column to save memory
df = df.drop("mask_ue")
# Collect lazy frame (execute everything)
df = df.collect()

# Set analysis confidence to zero for all rows with nonphysical transition locations 
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

print("Dataset:")
print(df)
# print("Dataset statistics:")
# print(df.describe())

### Shuffle the training set (deterministically)
df = df.sample(fraction=1, with_replacement=False, shuffle=True, seed=0)

print("At calculation now")

# Make the derivative dataset
# Apply to all rows
#derivatives_df = pl.DataFrame([compute_derivatives(row) for row in df.iter_rows(named=True)])
process_airfoil_batches_to_disk(df, derivatives_dir, batch_size=50_000) 

derivatives_files = sorted(derivatives_dir.glob("batch_*.parquet"))
derivatives_df = pl.concat([pl.read_parquet(f) for f in derivatives_files], how="vertical")

# Sort by original row index to ensure alignment
derivatives_df = derivatives_df.sort("row_index").drop("row_index")
print(f"Computed and loaded derivatives shape: {derivatives_df.shape}")

print(derivatives_df)

# Make the scaled datasets
df_inputs_scaled = pl.DataFrame(
    {
        **{f"s_kulfan_upper_{i}": df[f"kulfan_upper_{i}"] for i in range(8)},
        **{f"s_kulfan_lower_{i}": df[f"kulfan_lower_{i}"] for i in range(8)},
        "s_kulfan_LE_weight": df["kulfan_LE_weight"],
        "s_kulfan_TE_thickness": df["kulfan_TE_thickness"] * 50,
        "s_sin_2a": np.sind(2 * df["alpha"]),
        "s_cos_a": np.cosd(df["alpha"]),
        "s_1mcos2_a": 1 - np.cosd(df["alpha"]) ** 2,
        "s_Re": (np.log(df["Re"]) - 12.5) / 3.5,
        # No mach
        "s_n_crit": (df["n_crit"] - 9) / 4.5,
        "s_xtr_upper": df["xtr_upper"],
        "s_xtr_lower": df["xtr_lower"],
    }
)
print("Made scaled dataset")

mean_inputs_scaled_noderiv = np.mean(df_inputs_scaled.to_numpy(), axis=0)
cov_inputs_scaled_noderiv = np.cov(df_inputs_scaled.to_numpy(), rowvar=False)

# Compute the inverse of the covariance
inv_cov_inputs_scaled_noderiv = np.linalg.pinv(cov_inputs_scaled_noderiv)

# # Save everything to a .npz file
# np.savez(
#     "gen2_scaled_input_distribution_no_derivs.npz",
#     mean_inputs_scaled=mean_inputs_scaled_noderiv,
#     cov_inputs_scaled=cov_inputs_scaled_noderiv,
#     inv_cov_inputs_scaled=inv_cov_inputs_scaled_noderiv
# )

print("Calculated and saved input distribution information without derivatives.")

# Find index of "s_kulfan_TE_thickness"
insert_idx = df_inputs_scaled.columns.index("s_kulfan_TE_thickness") + 1
print("found insertion index")

# Slice dataframe in half to insert derivatives
before = df_inputs_scaled[:, :insert_idx]
after = df_inputs_scaled[:, insert_idx:]
print("cut dataframe at insertion index")

# Stack all together
df_inputs_scaled = pl.concat([before, derivatives_df, after], how="horizontal")
print("stacked inputs using pl.concat")

# di = df_inputs_scaled.describe()


df_outputs_scaled = pl.DataFrame(
    {
        "s_analysis_confidence": df["analysis_confidence"],
        "s_CL": 2 * df["CL"],
        "s_ln_CD": np.log(df["CD"]) / 2 + 2,
        "s_CM": 20 * df["CM"],
        "s_Top_Xtr": df["Top_Xtr"],
        "s_Bot_Xtr": df["Bot_Xtr"],
        **{
            f"s_upper_bl_ret_{i}": np.log10(
                np.abs(df[f"upper_bl_ue/vinf_{i}"])
                * df[f"upper_bl_theta_{i}"]
                * df["Re"]
                + 0.1
            )
            for i in range(Data.N)
        },
        **{
            f"s_upper_bl_H_{i}": np.log(df[f"upper_bl_H_{i}"] / 2.6)
            for i in range(Data.N)
        },
        **{
            f"s_upper_bl_ue/vinf_{i}": df[f"upper_bl_ue/vinf_{i}"]
            for i in range(Data.N)
        },
        **{
            f"s_lower_bl_ret_{i}": np.log10(
                np.abs(df[f"lower_bl_ue/vinf_{i}"])
                * df[f"lower_bl_theta_{i}"]
                * df["Re"]
                + 0.1
            )
            for i in range(Data.N)
        },
        **{
            f"s_lower_bl_H_{i}": np.log(df[f"lower_bl_H_{i}"] / 2.6)
            for i in range(Data.N)
        },
        **{
            f"s_lower_bl_ue/vinf_{i}": df[f"lower_bl_ue/vinf_{i}"]
            for i in range(Data.N)
        },
    }
)

# do = df_outputs_scaled.describe([0.01, 0.99])

### Split the dataset into train and test sets
test_train_split_index = int(len(df) * 0.95)

# Delete the variables
del df, derivatives_df, before, after

# Force garbage collection to free memory immediately
gc.collect()

print("Cleared df, derivatives_df, before, and after from memory.")

# df_train = df[:test_train_split_index]
# df_test = df[test_train_split_index:]
df_train_inputs_scaled = df_inputs_scaled[:test_train_split_index]
df_train_outputs_scaled = df_outputs_scaled[:test_train_split_index]
df_test_inputs_scaled = df_inputs_scaled[test_train_split_index:]
df_test_outputs_scaled = df_outputs_scaled[test_train_split_index:]
print("Splitting data between test and train sets has been completed")
print(f"The input training data is shaped as {df_train_inputs_scaled.describe()} ")
print(f"The output training data is shaped as {df_train_outputs_scaled.describe()} ")
print(f"The input test data is shaped as {df_test_inputs_scaled.describe()} ")
print(f"The output test data is shaped as {df_test_outputs_scaled.describe()} ")

# --------------------------------------------------
import gc
import polars as pl
import numpy as np
import sys

print("Scanning memory for large Polars DataFrames and NumPy arrays...")

# Force garbage collection to clear unused objects
gc.collect()

# Threshold for “large” objects (in bytes)
LARGE_THRESHOLD = 10_000  # ~1 GB

def format_bytes(n):
    """Human-readable MB string"""
    return f"{n / 1e6:.1f} MB"

# Combine globals and locals
objects_to_check = {**globals(), **locals()}

found = False
for name, obj in objects_to_check.items():
    try:
        if isinstance(obj, pl.DataFrame):
            n_rows, n_cols = obj.shape
            approx_bytes = n_rows * n_cols * 8  # float64 assumption
            if approx_bytes > LARGE_THRESHOLD:
                print(f"Polars DataFrame '{name}': {n_rows:,} rows x {n_cols:,} cols ~ {format_bytes(approx_bytes)}")
                found = True
        elif isinstance(obj, np.ndarray):
            n_bytes = obj.nbytes
            if n_bytes > LARGE_THRESHOLD:
                print(f"NumPy array '{name}': shape={obj.shape}, dtype={obj.dtype}, size={format_bytes(n_bytes)}")
                found = True
    except Exception:
        continue

if not found:
    print("No large Polars DataFrames or NumPy arrays found in memory.")
else:
    print("Consider deleting unneeded objects with `del var_name` + `gc.collect()` to free RAM.")


# Commented out below distribution metrics because analysis confidence works 
# with the distribution metrics calculated before the derivatives are added. 
cols = df_inputs_scaled.columns
p = len(cols)

# Compute mean of each column (eager DataFrame)
mean_inputs_scaled = df_inputs_scaled.select([pl.col(c).mean() for c in cols]).to_numpy().flatten()

# Compute covariance matrix
cov_exprs = []
for i in range(p):
    for j in range(i, p):
        col_i = pl.col(cols[i])
        col_j = pl.col(cols[j])
        cov_ij = ((col_i - mean_inputs_scaled[i]) * (col_j - mean_inputs_scaled[j])).mean()
        cov_exprs.append(cov_ij.alias(f"cov_{i}_{j}"))

cov_flat = df_inputs_scaled.select(cov_exprs).to_numpy().flatten()

# Reshape into 49×49 covariance matrix
cov_inputs_scaled = np.zeros((p, p))
k = 0
for i in range(p):
    for j in range(i, p):
        cov_inputs_scaled[i, j] = cov_flat[k]
        cov_inputs_scaled[j, i] = cov_flat[k]
        k += 1

# Compute pseudo-inverse
inv_cov_inputs_scaled = np.linalg.pinv(cov_inputs_scaled)

del df_inputs_scaled, df_outputs_scaled
gc.collect()
print("Deleted df_inputs_scaled and df_outputs_scaled from RAM" )

import psutil

mem = psutil.virtual_memory()

print(f"Total RAM: {mem.total / 1e9:.2f} GB")
print(f"Available RAM: {mem.available / 1e9:.2f} GB")
print(f"Used RAM: {mem.used / 1e9:.2f} GB")
print(f"RAM usage: {mem.percent}%")

# Save everything to a .npz file
# np.savez(
#     "gen2_scaled_input_distribution_K1-6.npz",
#     mean_inputs_scaled=mean_inputs_scaled,
#     cov_inputs_scaled=cov_inputs_scaled,
#     inv_cov_inputs_scaled=inv_cov_inputs_scaled
# )

#Saving for testing: 
# df_inputs_scaled.write_csv("test_load_data_new64_inputs.csv")
# derivatives_df.write_csv("test_load_data_new64_derivatives.csv")

print("----------- %s seconds -------" % (time.time() - start_time))

def make_data(row_index, df=df_test_inputs_scaled):
    row = df[row_index]
    return Data.from_vector(row[cols].to_numpy().flatten())


if __name__ == "__main__":
    # d = make_data(len(df_test_inputs_scaled) // 2, df_test_inputs_scaled)

    # Additional new scaled input distribution information for Avian training data
    _avian_scaled_input_distribution = dict(
        np.load("neuralfoil/nn_weights_and_biases/gen2_scaled_input_distribution.npz")
    )
    #Checked new code against old code that was run on Peter Sharpe data plus gen2 K1, K2, and K3
    d = _avian_scaled_input_distribution
    mean = d["mean_inputs_scaled"]
    cov = d["cov_inputs_scaled"]

    # direct_mean_inputs_scaled = np.mean(df_inputs_scaled.to_numpy(), axis=0)
    # direct_cov_inputs_scaled = np.cov(df_inputs_scaled.to_numpy(), rowvar=False)

    indices = list(range(0, 18)) + list(range(42, 49))
        
    mean_test = np.allclose(mean[indices], mean_inputs_scaled[indices], atol=1e-5)
    cov_test = np.allclose(cov, cov_inputs_scaled, atol=1e-5)
    # print(f" Means match: {mean_test}, Covariance match: {cov_test}")

    # mean_test = np.allclose(direct_mean_inputs_scaled, mean_inputs_scaled, atol=1e-5)
    # cov_test = np.allclose(direct_cov_inputs_scaled, cov_inputs_scaled, atol=1e-5)
    # print(f"Direct test Means match: {mean_test}, Covariance match: {cov_test}")

    max_diff = np.max(np.abs(mean - mean_inputs_scaled))
    print("Max absolute difference in mean:", max_diff)

    diff = np.abs(mean - mean_inputs_scaled)

    max_idx = np.argmax(diff)
    max_value = diff.flat[max_idx]   # or diff.reshape(-1)[max_idx]

    print("Max absolute difference in mean:", max_value)
    print("Index of max difference:", max_idx)

    max_diff = np.max(np.abs(cov - cov_inputs_scaled))
    print("Max absolute difference in covariance:", max_diff)
    
    print(f" Old Mean: {mean}")
    print(f" New Mean: {mean_inputs_scaled}")

    print("All done")


