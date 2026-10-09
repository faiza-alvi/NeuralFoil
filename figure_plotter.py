import numpy as np
import aerosandbox as asb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from aerosandbox.geometry.airfoil.airfoil_families import get_NACA_coordinates, get_kulfan_coordinates

# Get Kulfan parameters for NACA 16018
naca_16018_coords = np.loadtxt(r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\naca16018.dat", skiprows=1)
naca_16018 = asb.Airfoil(name="NACA 16018", coordinates=naca_16018_coords).normalize().to_kulfan_airfoil()

# get Kulfan parameters for Bird airfoil
puffinus_coords = np.loadtxt(r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\BirdAirfoils\Puffinus_tenuirostris_NCUnknown79_0.40.csv", delimiter=",")
puffinus = asb.Airfoil(name="Puffinus tenuirostris, 0.40", coordinates=puffinus_coords).normalize().to_kulfan_airfoil()

# get Kulfan parameters for 
naca_2412_coords = get_NACA_coordinates(name="naca2412")
naca_2412 = asb.Airfoil(name="NACA 2412", coordinates=naca_2412_coords).normalize().to_kulfan_airfoil()

parent_airfoils = [naca_16018, puffinus, naca_2412]

# combination code
n_airfoils_to_combine = 3
np.random.seed(34)

slices = np.random.rand(n_airfoils_to_combine - 1)
slices = np.sort(slices)
slices = np.concatenate([[0], slices, [1]])
weights = np.diff(slices)  # result is N random numbers in [0, 1] that sum to 1

af = asb.KulfanAirfoil(
    name="Reconstructed Airfoil",
    upper_weights=np.dot(
        weights,
        [parent_airfoil.upper_weights for parent_airfoil in parent_airfoils],
    ),
    lower_weights=np.dot(
        weights,
        [parent_airfoil.lower_weights for parent_airfoil in parent_airfoils],
    ),
    leading_edge_weight=np.dot(
        weights,
        [
            parent_airfoil.leading_edge_weight
            for parent_airfoil in parent_airfoils
        ],
    ),
    TE_thickness=np.dot(
        weights,
        [parent_airfoil.TE_thickness for parent_airfoil in parent_airfoils],
    ),
)
af = af.scale(1, np.random.lognormal(0, 0.25))

# Figure stuff
# Names for top row
names = [
    "NACA 16-018",
    f"Puffinus tenuirostris, 40% along the span",
    "NACA 2412"
]
colors = [
    "#0D4079",  # blue
    "#3B7D23",  # red
    "#005555"   # green
]

# -------------------------------------------------------------------
# Figure layout
# -------------------------------------------------------------------

fig = plt.figure(
    figsize=(14, 6),
    constrained_layout=True
)

gs = GridSpec(2, 8,
    figure=fig,
    height_ratios=[1, 1.2]
)
fig.set_constrained_layout_pads(
    w_pad=0.02,
    h_pad=0.0,
    wspace=0.02,
    hspace=0.0
)
# -------------------------------------------------------------------
# Top row: 3 separate airfoil plots
# -------------------------------------------------------------------

top_axes = [
    fig.add_subplot(gs[0, 1:3]),
    fig.add_subplot(gs[0, 3:5]),
    fig.add_subplot(gs[0, 5:7]),
]

airfoils = [
    (naca_16018_coords[:,0], naca_16018_coords[:, 1]),
    (puffinus_coords[:, 0], puffinus_coords[:, 1]),
    (naca_2412_coords[:, 0], naca_2412_coords[:, 1]),
]

for ax, (x, z), name, color in zip(top_axes, airfoils, names, colors):

    ax.plot(x, z, linewidth=2, color=color)
    ax.fill(x, z, color=color, alpha=0.25)

    ax.set_title(name, fontsize=14, pad=10)

    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 0.4)

    # Clean appearance
    ax.axis('off')

# -------------------------------------------------------------------
# Bottom-left: overlapping airfoils
# -------------------------------------------------------------------

ax_overlap = fig.add_subplot(gs[1, 1:4])

ax_overlap.plot(naca_16018_coords[:, 0], naca_16018_coords[:, 1], linewidth=2, color = colors[0])
ax_overlap.plot(puffinus_coords[:, 0], puffinus_coords[:, 1], linewidth=2, color=colors[1])
ax_overlap.plot(naca_2412_coords[:, 0], naca_2412_coords[:, 1], linewidth=2, color=colors[2])

ax_overlap.fill(naca_16018_coords[:, 0], naca_16018_coords[:, 1], color = colors[0], alpha=0.25)
ax_overlap.fill(puffinus_coords[:, 0], puffinus_coords[:, 1], color=colors[1], alpha=0.25)
ax_overlap.fill(naca_2412_coords[:, 0], naca_2412_coords[:, 1], color=colors[2], alpha=0.25)

ax_overlap.set_xlabel("x/c")
ax_overlap.set_ylabel("z/c")

ax_overlap.set_aspect('equal')

ax_overlap.set_xlim(-0.1, 1.1)
ax_overlap.set_ylim(-0.20, 0.30)

# Remove top/right box borders
ax_overlap.spines['top'].set_visible(False)
ax_overlap.spines['right'].set_visible(False)

# Keep only left/bottom axes
ax_overlap.yaxis.set_ticks_position('left')
ax_overlap.xaxis.set_ticks_position('bottom')
ax_overlap.set_title("Input Airfoils", fontsize=14)

# -------------------------------------------------------------------
# Bottom-right: synthetic airfoil
# -------------------------------------------------------------------

ax_syn = fig.add_subplot(gs[1, 4:7])

ax_syn.plot(af.coordinates[:, 0], af.coordinates[:, 1], linewidth=2, color="#0faeae")
ax_syn.fill(af.coordinates[:, 0], af.coordinates[:, 1], color="#0faeae", alpha = 0.25)

ax_syn.set_xlabel("x/c")
ax_syn.set_ylabel("z/c")

ax_syn.set_aspect('equal')

ax_syn.set_xlim(-0.1, 1.1)
ax_syn.set_ylim(-0.2, 0.3)

ax_syn.set_title("Synthetic Airfoil", fontsize=14)

# Remove top/right box borders
ax_syn.spines['top'].set_visible(False)
ax_syn.spines['right'].set_visible(False)


# Keep only left/bottom axes
ax_syn.yaxis.set_ticks_position('left')
ax_syn.xaxis.set_ticks_position('bottom')

# -------------------------------------------------------------------
# Optional overall title
# -------------------------------------------------------------------

# fig.suptitle("Airfoil Comparison", fontsize=16)

plt.show()
# fig.savefig("synthesized_airfoils.png", dpi=300)

# -------------------------------------------------------------
# Figure 3: Loss plots 
# --------------------------------------------------------------
import re
import matplotlib.pyplot as plt

# regex to capture epoch, train loss, test loss
pattern = re.compile(
    r"Epoch:\s*(\d+)\s*\|\s*Train Loss:\s*([\d.eE+-]+)\s*\|\s*Test Loss:\s*([\d.eE+-]+)"
)


train_losses_128 = []
valid_losses_128 = []
train_losses_256 = []
valid_losses_256 = []
train_losses_512 = []
valid_losses_512 = []
train_losses_1024 = []
valid_losses_1024 = []



# List of log files
log_files = [r"avian_gen2_128.log"]
for file in log_files:
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                train_losses_128.append(float(match.group(2)))
                valid_losses_128.append(float(match.group(3)))

log_files = [r"avian_gen2_256.log", r"avian_gen2_256-2.log"]
for file in log_files:
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                train_losses_256.append(float(match.group(2)))
                valid_losses_256.append(float(match.group(3)))

log_files = [r"avian_gen2_512.log", r"avian_gen2_512-2.log", r"avian_gen2_512-3.log"]
for file in log_files:
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                train_losses_512.append(float(match.group(2)))
                valid_losses_512.append(float(match.group(3)))

log_files = [r"avian_gen2_1024.log"]
for file in log_files:
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                train_losses_1024.append(float(match.group(2)))
                valid_losses_1024.append(float(match.group(3)))


# Plot
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

losses = [
    (train_losses_128, valid_losses_128),
    (train_losses_256, valid_losses_256),
    (train_losses_512, valid_losses_512),
    (train_losses_1024, valid_losses_1024)
]

for ax, (train, valid) in zip(axs.flat, losses):
    ax.plot(train, lw=2, color="#0faeae", label="Train Loss")
    ax.plot(valid, lw=2, color="#e9692c", label="Validation Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("Loss", fontsize=18)
    ax.tick_params(labelsize=16, length=5)
    ax.spines[["top", "right"]].set_visible(False)

# Labels outside plots
for ax, label in zip(axs.flat, ["A", "B", "C", "D"]):
    ax.text(-0.12, 1.02, label, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="bottom", ha="right")

# Shared legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center left",
           bbox_to_anchor=(0.90, 0.5), frameon=False, fontsize=16)

plt.tight_layout(rect=[0.03, 0, 0.87, 1])
fig.savefig("C:/Users/booki/Documents/SciTech 2027/Figures/AvianFoil_loss_curves.svg", format="svg", transparent=True, bbox_inches="tight")
plt.show()


# plt.plot(train_losses_128, label="Train Loss", linewidth=2, color="#0faeae")
# plt.plot(valid_losses_128, label="Validation Loss", linewidth=2, color="#e9692c")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("AvianFoil Batch Size 128 Model Loss")
# plt.yscale("log")   # if you want log scale like before
# plt.legend()
# plt.show()