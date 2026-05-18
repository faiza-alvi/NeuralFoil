import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.colors import PowerNorm
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors

# custom_cmap = mcolors.LinearSegmentedColormap.from_list(
#     "orange_teal",
#     ["#e9692c", "#ffffff", "#0faeae"],  # white midpoint keeps 0 neutral
#     N=256
# )
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "white_orange_teal",
    [
        (0.0, "#ffffff"),  # zero -> gray
        (0.001, "#e9692c"),  # immediately transition to orange
        (1.0, "#0faeae"),  # high values -> teal
    ],
    N=256
)

folder_path = r'/home/faiza/Documents/Gen2TrainingAirfoils' #CHANGE FILE LOCATION
import os
complete_data = glob.glob(os.path.join(folder_path, "*.csv"))
complete_x = [] #angle of attack
complete_y = [] #Re

#loop through each file
for file in complete_data:   
    data = pd.read_csv(file)
    print(f"Read the file {file}")
    x = data.iloc[:, 18] #take entire data from columns
    y = data.iloc[:, 19]
    complete_x.append(x) #combine x values into complete_x
    complete_y.append(y) #combine y values into complete_y

# #combine all data into single arrays
# complete_x = pd.concat(complete_x)
# complete_y = pd.concat(complete_y)

# #heatmap setup
# counts, xedges, yedges = np.histogram2d(complete_x, complete_y, bins=20)
# cmap = custom_cmap
# norm = PowerNorm(gamma=0.5, vmin=counts.min(), vmax=counts.max())
# #gamma < 1 means smaller values have more weight
# #gamma > 1 means larger values have more weight

# #plot
# plt.figure(figsize=(8,6))
# plt.pcolormesh(xedges, yedges, counts.T, cmap=cmap, norm=norm)
# plt.colorbar(label='Number of points')
# plt.title("Angle of Attack vs Reynolds Number")
# plt.xlabel("Angle of Attack")
# plt.ylabel("Reynolds Number")
# plt.grid(True)
# plt.show()

# combine all data into single arrays
complete_x = pd.concat(complete_x)
complete_y = pd.concat(complete_y)

# ---- Create bins ----
x_bins = 20  # keep linear bins for x

# Log-spaced bins for Reynolds number (y)
y_min = complete_y.min()
y_max = complete_y.max()
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), 20)

# ---- Compute histogram ----
counts, xedges, yedges = np.histogram2d(
    complete_x,
    complete_y,
    bins=[x_bins, y_bins]
)

# ---- Colormap normalization ----
cmap = custom_cmap
norm = PowerNorm(gamma=0.5, vmin=counts.min(), vmax=counts.max())

# ---- Plot ----
plt.figure(figsize=(6,5))
plt.pcolormesh(xedges, yedges, counts.T, cmap=cmap, norm=norm)

plt.yscale("log")  # <-- THIS makes axis logarithmic

plt.colorbar(label='Number of points')
plt.title("Angle of Attack vs Reynolds Number")
plt.xlabel("Angle of Attack")
plt.ylabel("Reynolds Number (logarithmic)")
plt.grid(True, which="both")
plt.show()
