import matplotlib.pyplot as plt
import numpy as np

# Assuming the data is stored in a dictionary called data
data = {
    "contrast": {
        0: {1: 15971.692556634303},
        45: {1: 19113.56338598623},
        90: {1: 13957.137259413172},
        270: {1: 15971.692556634303},
    },
    "homogeneity": {
        0: {1: 0.7543799010144512},
        45: {1: 0.7060627535757047},
        90: {1: 0.7853606671267928},
        270: {1: 0.7543799010144512},
    },
    "energy": {
        0: {1: 0.7034156115291843},
        45: {1: 0.6872205717140731},
        90: {1: 0.7152112943851201},
        270: {1: 0.7034156115291843},
    },
    "correlation": {
        0: {1: 0.232218300864926},
        45: {1: 0.08197098022965997},
        90: {1: 0.3290355126634479},
        270: {1: 0.232218300864926},
    },
}

# Create a figure with 4 subplots, one for each feature
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Loop through the features and plot the histograms
for i, feature in enumerate(data.keys()):
    # Get the row and column index of the subplot
    row = i // 2
    col = i % 2

    # Get the values and angles for the feature
    values = list(data[feature].values())
    angles = list(data[feature].keys())

    # Plot the histogram on the subplot
    axes[row, col].bar(angles, values, width=20)
    axes[row, col].set_title(feature)
    axes[row, col].set_xlabel("Angle")
    axes[row, col].set_ylabel("Value")

# Adjust the layout and show the figure
plt.tight_layout()
plt.show()
