# import csv
# import aerosandbox as asb
# import aerosandbox.numpy as np
# from aerosandbox import Airfoil
# from typing import List
# import time
# from neuralfoil._basic_data_type import Data
# from pathlib import Path

#testing plots, commented out below 
"""
af=Airfoil("naca2412")
alphas = (
            np.linspace(-15, 15, 7)
            + np.random.uniform(-2.5, 2.5)
            + 2.5 * np.random.randn()
        )
Re = float(10 ** (5.5 + 1.5 * np.random.randn()))

n_crit = np.random.uniform(0, 18)
if np.random.rand() < 0.8:
    xtr_upper = 1
else:
    xtr_upper = np.random.uniform(0, 1)
if np.random.rand() < 0.8:
    xtr_lower = 1
else:
    xtr_lower = np.random.uniform(0, 1)

datas = Data.from_xfoil(
            airfoil=af,
            alphas=alphas,
            Re=Re,
            mach=0,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
            timeout=60,
            max_iter=200,
            # xfoil_command="/home/faiza/Documents/xfoil"            
        )

import pandas as pd

for i, data in enumerate(datas):
    # Convert to DataFrame; assuming to_vector() returns list or dict
    df = pd.DataFrame([data.to_vector()])  # single-row DataFrame
    
    if i == 0:
        df.to_csv("testrun.csv", index=False)
    else:
        df.to_csv("testrun.csv", mode='a', header=False, index=False)

"""
# ----------------------------------------------
# Code for plotting training losses
import re
import matplotlib.pyplot as plt

# regex to capture epoch, train loss, test loss
pattern = re.compile(
    r"Epoch:\s*(\d+)\s*\|\s*Train Loss:\s*([\d.eE+-]+)\s*\|\s*Test Loss:\s*([\d.eE+-]+)"
)

epochs = []
train_losses = []
test_losses = []

# Path for xxxlarge: C:\Users\booki\Documents\BIRD Lab\Airfoil Project\NeuralFoil\training\log.log-25543001
# path for avian: C:\Users\booki\Documents\BIRD Lab\Airfoil Project\NeuralFoil\training\avian.log

# List of log files
log_files = [
    r"avian_gen2_256.log",
    r"avian_gen2_256-2.log"
    # r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\NeuralFoil\avian-v3_gen2_pt1.log",
    # r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\NeuralFoil\avian-v3_gen2_pt2.log", 
    # r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\NeuralFoil\avian-v3_gen2_pt3.log",
    # r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\NeuralFoil\avian-v3_gen2_pt4.log"
    # r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\NeuralFoil\training\log.log-25543001"
]

for file in log_files:
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                test_losses.append(float(match.group(3)))

train_sub = train_losses[-50:]
test_sub = test_losses[-50:]
train_conv_error = max(train_sub) - min(train_sub)
test_conv_error = max(test_sub) - min(test_sub)
print(f"The training convergence error is {train_conv_error}")
print(f"The validation convergence error is {test_conv_error}")
print(f"This is comparing over the last {len(train_sub)} epochs")
print(f"The two loss values are approximately {max(test_sub) - min(train_sub)} apart.")

# Plot
plt.plot(train_losses, label="Train Loss", linewidth=2, color="#0faeae")
plt.plot(test_losses, label="Validation Loss", linewidth=2, color="#e9692c")
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.xlim(0,250)
#plt.ylim(0, 2)
plt.yscale("log")   # if you want log scale like before
plt.legend()
# plt.title("AvianFoil Preliminary Model Loss")
plt.show()

# ------------------------------------------------------------
# Code to test covariances 

# import numpy as np

# # Load the .npz file
# data = np.load('neuralfoil/nn_weights_and_biases/avian_scaled_input_distribution.npz')

# # List all arrays stored in the file
# print(data.files)

# # Loop through all arrays in the file
# for name in data.files:
#     array = data[name]
#     print(f"Array name: {name}")
#     print(f"Shape: {array.shape}")
#     #print(f"Data preview:\n{array}\n")

# # Extract the arrays
# cov = data['cov_inputs_scaled']
# inv_cov = data['inv_cov_inputs_scaled']
# mean = data['mean_inputs_scaled']

# new_inv = np.linalg.pinv(cov)

# # Multiply the two matrices
# product = cov @ inv_cov  # matrix multiplication
# print(np.allclose(product, np.eye(cov.shape[0])))

# new_product = cov @ new_inv  # matrix multiplication
# print("New pinv")
# print(np.allclose(new_product, np.eye(cov.shape[0])))

# condition_number = np.linalg.cond(cov)
# print("Condition number:", condition_number)

# pinv = np.linalg.pinv(cov)

# product = cov @ pinv
# print("Symmetric:", np.allclose(product, product.T, atol=1e-8))
# print("Idempotent:", np.allclose(product @ product, product, atol=1e-8))
# print("Rank:", np.linalg.matrix_rank(cov), "/", cov.shape[0])

# # Testing with a generated input vector for NACA 2412
# import aerosandbox as asb
# import aerosandbox.numpy as np
# from aerosandbox.geometry.airfoil.airfoil_families import (
#     get_NACA_coordinates,
#     get_kulfan_parameters,
# )
# from scipy.special import comb


# # ============================================================
# # Derivative helper (with small bug fixes: np.length -> len)
# # ============================================================
# def derivatives_at_nodes(
#     lower_weights,
#     upper_weights,
#     leading_edge_weight,
#     TE_thickness,
# ):
#     N1 = 0.5
#     N2 = 1.0

#     n_weights_per_side = len(lower_weights)

#     # Nodes for derivative evaluation
#     x_nodes = np.linspace(0, 1, n_weights_per_side)
#     x_nodes = x_nodes[1:-1]   # interior nodes only
#     n_nodes = len(x_nodes)

#     N = n_weights_per_side - 1
#     K = comb(N, np.arange(N + 1))    # Bernstein coefficients

#     dims = (n_weights_per_side, n_nodes)

#     def wide(v):
#         return np.tile(np.reshape(v, (1, dims[1])), (dims[0], 1))

#     def tall(v):
#         return np.tile(np.reshape(v, (dims[0], 1)), (1, dims[1]))

#     # Exponents
#     p = np.arange(N1, N1 + N + 1)
#     q = N - np.arange(N + 1) + N2

#     p_1 = p - 1
#     q_1 = q - 1
#     p_2 = p - 2
#     q_2 = q - 2
#     q_2[q_2 < 0] = 0

#     slopes_matrix = (
#         tall(K) * tall(p) * wide(x_nodes) ** tall(p_1) * wide(1 - x_nodes) ** tall(q)
#         - tall(K) * tall(q) * wide(x_nodes) ** tall(p) * wide(1 - x_nodes) ** tall(q_1)
#     )

#     curvature_matrix = (
#         tall(K) * tall(p) * tall(p_1) * wide(x_nodes) ** tall(p_2) * wide(1 - x_nodes) ** tall(q)
#         - 2 * tall(K) * tall(q) * tall(p) * wide(x_nodes) ** tall(p_1) * wide(1 - x_nodes) ** tall(q_1)
#         + tall(K) * tall(q) * tall(q_1) * wide(x_nodes) ** tall(p) * wide(1 - x_nodes) ** tall(q_2)
#     )

#     lowerslope = slopes_matrix.T @ lower_weights
#     lowercurve = curvature_matrix.T @ lower_weights
#     upperslope = slopes_matrix.T @ upper_weights
#     uppercurve = curvature_matrix.T @ upper_weights

#     # Leading edge modification
#     m_upper = len(upper_weights) + 0.5
#     m_lower = len(lower_weights) + 0.5

#     LE_upper_slope = (
#         leading_edge_weight * ((1 - x_nodes) ** m_upper)
#         - leading_edge_weight * m_upper * x_nodes * ((1 - x_nodes) ** (m_upper - 1))
#     )
#     LE_lower_slope = (
#         leading_edge_weight * ((1 - x_nodes) ** m_lower)
#         - leading_edge_weight * m_lower * x_nodes * ((1 - x_nodes) ** (m_lower - 1))
#     )

#     LE_upper_curve = (
#         -2 * leading_edge_weight * m_upper * ((1 - x_nodes) ** (m_upper - 1))
#         + leading_edge_weight * m_upper * (m_upper - 1)
#         * x_nodes * ((1 - x_nodes) ** (m_upper - 2))
#     )
#     LE_lower_curve = (
#         -2 * leading_edge_weight * m_lower * ((1 - x_nodes) ** (m_lower - 1))
#         + leading_edge_weight * m_lower * (m_lower - 1)
#         * x_nodes * ((1 - x_nodes) ** (m_lower - 2))
#     )

#     upperslope += LE_upper_slope
#     lowerslope += LE_lower_slope
#     uppercurve += LE_upper_curve
#     lowercurve += LE_lower_curve

#     # TE thickness effect (constant slope shift)
#     upperslope += TE_thickness / 2
#     lowerslope -= TE_thickness / 2

#     return {
#         "upper_first_der": upperslope,
#         "lower_first_der": lowerslope,
#         "upper_second_der": uppercurve,
#         "lower_second_der": lowercurve,
#     }


# # ============================================================
# # Generate Kulfan parameters for NACA 2412
# # ============================================================
# coords = get_NACA_coordinates("naca2412", n_points_per_side=200)
# kulfan = get_kulfan_parameters(
#     coordinates=coords,
#     n_weights_per_side=8,
#     N1=0.5,
#     N2=1.0,
#     use_leading_edge_modification=True,
#     normalize_coordinates=True,
# )

# # ============================================================
# # Compute derivatives
# # ============================================================
# derivs = derivatives_at_nodes(
#     lower_weights=kulfan["lower_weights"],
#     upper_weights=kulfan["upper_weights"],
#     leading_edge_weight=kulfan["leading_edge_weight"],
#     TE_thickness=kulfan["TE_thickness"],
# )

# # ============================================================
# # Form the input vector x
# # ============================================================
# Re = 100_000
# alpha = 2.0
# ncrit = 9.0
# xtr_upper = 1.0
# xtr_lower = 1.0

# x = np.array([
#     *kulfan["upper_weights"],
#     *kulfan["lower_weights"],
#     kulfan["leading_edge_weight"],
#     kulfan["TE_thickness"] * 50,           # your scaling
#     *derivs["upper_first_der"],
#     *derivs["lower_first_der"],
#     *derivs["upper_second_der"],
#     *derivs["lower_second_der"],
#     np.sind(2 * alpha),
#     np.cosd(alpha),
#     1 - np.cosd(alpha) ** 2,
#     (np.log(Re) - 12.5) / 3.5,
#     # No Mach
#     (ncrit - 9) / 4.5,
#     xtr_upper,
#     xtr_lower,
# ])

# print("Final input vector x shape:", x.shape)
# # print("x =", x)


# x_centered = x - mean

# # squared Mahalanobis distance
# D2 = float(x_centered @ inv_cov @ x_centered)

# print("Squared Mahalanobis distance with original 49x49 inv:", D2)

# # Now using pinv instead of inv_cov
# pinv_dist = float(x_centered @ pinv @ x_centered)
# print("Squared Mahalanobis distance with 49x49 pinv: ", pinv_dist )

# #Now with subsetting
# remove_index = np.arange(18, 42)
# # Indices to KEEP
# keep = np.setdiff1d(np.arange(cov.shape[0]), remove_index)

# # Remove rows and columns
# cov_reduced = cov[np.ix_(keep, keep)]
# inv_cov_reduced = np.linalg.inv(cov_reduced)

# x_centered_reduced = x_centered[np.r_[0:18, 42:49]]

# inv_dist_red = float(x_centered_reduced @ inv_cov_reduced @ x_centered_reduced)
# print("Squared Mahalanobis distance with reduced 25x25 dimension and inverse: ", inv_dist_red )

# pinv_reduced = np.linalg.pinv(cov_reduced)
# print(pinv_reduced.shape)
# pinv_dist_red = float(x_centered_reduced @ pinv_reduced @ x_centered_reduced)
# print("Squared Mahalanobis distance with reduced 25x25 dimension and pinv: ", pinv_dist_red )
# print("Rank of pinv reduced:", np.linalg.matrix_rank(pinv_reduced), "/", pinv_reduced.shape[0])

# # Multiply the two matrices
# product = cov_reduced @ pinv_reduced  # matrix multiplication
# print("Is reduced matrix an inverse?: ", np.allclose(product, np.eye(cov_reduced.shape[0])))

# data2 = np.load('neuralfoil/nn_weights_and_biases/scaled_input_distribution.npz')
# cov2 = data2['cov_inputs_scaled']
# inv_cov2 = data2['inv_cov_inputs_scaled']
# mean2 = data2['mean_inputs_scaled']
# orig_dist = float(x_centered_reduced @ inv_cov2 @ x_centered_reduced)
# print("Mahalanobis distance for original database:", orig_dist)

# frobenius_distance = np.linalg.norm(cov_reduced - cov2, ord='fro')
# relative_frobenius_distance = frobenius_distance / np.linalg.norm(cov_reduced, ord='fro')

# print("Frobenius distance:", frobenius_distance)
# print("Relative Frobenius distance:", relative_frobenius_distance)
# print("Percent difference:", relative_frobenius_distance * 100, "%")

# # w1, v1 = np.linalg.eigh(cov_reduced)
# # w2, v2 = np.linalg.eigh(cov2)

# # print("Eigenvalues cov1:", w1)
# # print("Eigenvalues cov2:", w2)

# data3 = np.load('avian_scaled_input_distribution_no_derivs.npz')
# cov3 = data3['cov_inputs_scaled']
# inv_cov3 = data3['inv_cov_inputs_scaled']
# mean3 = data3['mean_inputs_scaled']
# no_deriv_dist = float(x_centered_reduced @ inv_cov3 @ x_centered_reduced)
# print("Mahalanobis distance for avian database without derivatives:", no_deriv_dist)

# data4 = np.load(r"neuralfoil\nn_weights_and_biases\gen2_scaled_input_distribution_no_derivs.npz")
# cov4 = data4['cov_inputs_scaled']
# inv_cov4 = data4['inv_cov_inputs_scaled']
# mean4 = data4['mean_inputs_scaled']
# no_deriv_dist_Peter = float(x_centered_reduced @ inv_cov4 @ x_centered_reduced)
# print("Mahalanobis distance for Gen2's database without derivatives:", no_deriv_dist_Peter)

# data4 = np.load('gen2_scaled_input_distribution.npz')
# cov4 = data4['cov_inputs_scaled']
# inv_cov4 = np.linalg.pinv(cov4)#data4['inv_cov_inputs_scaled']
# mean4 = data4['mean_inputs_scaled']
# no_deriv_dist_Peter = float(x_centered @ inv_cov4 @ x_centered)
# print("Mahalanobis distance for Gen2's database with derivatives:", no_deriv_dist_Peter)


# # Suppose cov1 and cov2 are your covariance matrices
# eig1 = np.linalg.eigvalsh(cov_reduced)  # sorted ascending
# eig2 = np.linalg.eigvalsh(cov2)
# eig3 = np.linalg.eigvalsh(cov3)

# # # Plot on a semilog scale to see small eigenvalues clearly
# # plt.figure(figsize=(8,5))
# # plt.semilogy(eig1, '.-', label='Avian Covariance - Reduced Matrix')
# # plt.semilogy(eig2, 's-', label='Original Covariance')
# # plt.semilogy(eig3, ':', label='Avian Covariance - no derivatives')
# # plt.xlabel('Eigenvalue index')
# # plt.ylabel('Eigenvalue (log scale)')
# # plt.title('Comparison of Covariance Eigenvalues')
# # plt.grid(True, which='both', ls='--', alpha=0.5)
# # plt.legend()
# # plt.show()

# --------------------------------------------------------
# Code to check that two matrices (saved as .csv) are the same 

# import numpy as np
# import pandas as pd

# # Read csv files
# mat1 = pd.read_csv("test_load_data_original_inputs.csv",skiprows=1, header=None).to_numpy()
# mat2 = pd.read_csv("test_load_data_new64_inputs.csv", skiprows=1, header=None).to_numpy()
# print(mat1.shape)
# print(mat2.shape)
# # Check if matrices are identical
# are_equal = np.allclose(mat1, mat2, atol=1e-5)

# print("Matrices are identical:", are_equal)

# max_diff = np.max(np.abs(mat1 - mat2))
# print("Max absolute difference:", max_diff)

# --------------------------------------------------------
# Code to plot the airfoils from one training dataset 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import aerosandbox as asb
from aerosandbox.geometry.airfoil.airfoil_families import (
    get_kulfan_coordinates,
)

# # --- Load CSV ---
# df = pd.read_csv("your_file.csv")

# # --- Extract first 18 columns as Kulfan parameters ---
# kulfan_params = df.iloc[:, :18].values  # shape: (num_airfoils, 18)

# # --- Plot setup ---
# plt.figure(figsize=(8, 4))

# # --- Loop through each airfoil ---
# for i, params in enumerate(kulfan_params):
#     coords = get_kulfan_coordinates(
#         kulfan_parameters=params,
#         n_points_per_side=100  # adjust resolution if needed
#     )
    
#     x = coords[:, 0]
#     y = coords[:, 1]
    
#     plt.plot(x, y, alpha=0.5)

# # --- Formatting ---
# plt.gca().set_aspect('equal', adjustable='box')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Gen2 Training Airfoils")
# plt.grid(True)

# plt.show()
