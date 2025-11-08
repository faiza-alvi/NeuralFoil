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

# with open(r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\NeuralFoil\training\avian_9-16-25.log", "r", encoding="utf-8", errors="ignore") as f:
#     for line in f:
#         match = pattern.search(line)
#         if match:
#             epochs.append(int(match.group(1)))
#             train_losses.append(float(match.group(2)))
#             test_losses.append(float(match.group(3)))

# # Plot
# plt.plot(train_losses, label="Train Loss", linewidth=1)
# plt.plot(test_losses, label="Test Loss", linewidth=1)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.yscale("log")   # if you want log scale like before
# plt.legend()
# plt.title("Training Progress")
# plt.show()

import numpy as np

# Load the .npz file
data = np.load('neuralfoil/nn_weights_and_biases/scaled_input_distribution.npz')

# List all arrays stored in the file
print(data.files)

# Loop through all arrays in the file
for name in data.files:
    array = data[name]
    print(f"Array name: {name}")
    print(f"Shape: {array.shape}")
    print(f"Data preview:\n{array}\n")