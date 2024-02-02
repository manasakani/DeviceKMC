import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Define a dictionary to store the timing information
timing_data = {
    "KMC step count": [],
    "Z - calculation time - charge [s]": [],
    "Z - calculation time - dissipated power [s]": [],
    "Z - calculation time - potential from boundaries [s]": [],
    "Z - calculation time - potential from charges [s]": [],
    "Z - calculation time - kmc events [s]": [],
    "Z - calculation time - KMC superstep [s]": [],
}

# Read the log file and extract timing information
with open("output.txt", "r") as file:
    lines = file.readlines()
    current_kmc_step = None

    for line in lines:
        if "KMC step count" in line:
            current_kmc_step = int(line.split(":")[-1].strip())
        elif "Z - calculation time" in line:
            key, value = map(str.strip, re.split(r':\s+', line, maxsplit=1))
            if key in timing_data:
                timing_data[key].append(float(value))
                timing_data["KMC step count"].append(current_kmc_step)

# Plot box plots for each module
fig, ax = plt.subplots(figsize=(8, 8))

# Create a list of lists for boxplot
data = [timing_data[key] for key in timing_data.keys() if key != "KMC step count"]
labels = [key.replace("Z - calculation time - ", "").replace("[s]", "") for key in timing_data.keys() if key != "KMC step count"]
labels = ["Total - KMC superstep" if label == "KMC superstep " else label for label in labels]

# Set up different colors for each box
box_colors = plt.cm.viridis(np.linspace(0, 1, len(data)))

medianprops = dict(linestyle='-', linewidth=2, color='black')
flierprops = dict(marker='o', markersize=8, markerfacecolor='none', markeredgecolor=box_colors)

# Plot the boxplot with different colors for each box
bp = ax.boxplot(data, medianprops=medianprops, patch_artist=True, labels=labels, flierprops=flierprops, boxprops=dict(alpha=0.1))

# Set box colors
for box, color in zip(bp['boxes'], box_colors):
    box.set_facecolor(color)
    box.set_edgecolor(color)

# Set the y-axis to log scale
ax.set_yscale('log')
ax.set_ylim(1e-6, 1e2)

# Increase font size
ax.tick_params(axis='both', which='major', labelsize=18)

# Remove x-axis labels
ax.set_xticklabels([])

# Color the outlier edges with the corresponding box colors
for flier, color in zip(bp['fliers'], box_colors):
    flier.set(markerfacecolor='none', markeredgecolor=color)

# Create a colored legend
legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(box_colors, labels)]
ax.legend(handles=legend_patches, loc='upper right', fontsize=16, bbox_to_anchor=(1.0, -0))

# Plot lines for each measurement with alpha corresponding to the module position
for x, (values, color) in enumerate(zip(data, box_colors), start=1):
    track = 0
    for value in values:
        # ax.plot(x, value, 'o', color=color, alpha = x / len(data), markeredgewidth=0)
        ax.plot(x, value, '.', markersize = 10, color=color, alpha = track / max(timing_data["KMC step count"]), markeredgewidth=0)
        track+=1

# # Plot lines for each measurement with alpha corresponding to KMC step number
# for kmc_step, color, values in zip(timing_data["KMC step count"], box_colors, data):
#     module_number = 1  # Initialize module number
#     for value in values:
#         ax.plot(module_number, value, 'o', color=color, alpha=kmc_step / max(timing_data["KMC step count"]), markeredgewidth=0)
#     module_number += 1  # Increment module number
#     print(module_number)


plt.ylabel("Time (s)", fontsize=18)

# Save the plot as an image
plt.savefig('module_timings_boxplot.png', bbox_inches='tight', dpi=300)
plt.show()
