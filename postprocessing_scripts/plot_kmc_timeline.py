import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde


def extract_time_points(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        time_points = [float(line.split()[3])*1e9 for line in lines if 'KMC time is:' in line]
    return time_points

# Function to plot time points on a timeline with translucent circles
def plot_timeline(time_points):
    plt.figure(figsize=(10, 5))
        
    # Perform kernel density estimation (KDE) to estimate the local density
    kde = gaussian_kde(time_points)
    density = kde(time_points)

    # Manually set colors for each vertical tick based on the local density
    norm = plt.Normalize(np.min(density), np.max(density))
    colors = plt.cm.viridis(norm(density))

    # Create a ScalarMappable to map the colors
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])

    # Plot colored vertical ticks
    for time, color in zip(time_points, colors):
        plt.vlines(time, 0, 1, colors=color, alpha=0.05, linewidth=2)

        # plt.vlines(time, 0, 1, colors=color, alpha=0.5, linewidth=2)

    # Set x-axis to linear scale with 5 ticks
    plt.xscale('linear')
    tick_positions = np.linspace(min(time_points), max(time_points), 5)
    # plt.xticks(tick_positions, [f'{tick:.2e}' for tick in tick_positions])
    plt.xticks(tick_positions, ['{:.2f}'.format(tick) for tick in tick_positions], fontsize=12)  # Format x-axis ticks as decimals


    plt.xlabel('Physical time (ns)', fontsize=14)
    plt.title('KMC superstep time distribution', fontsize=14)
    plt.yticks([])  # Hide y-axis

    # Add colorbar legend
    cbar = plt.colorbar(sm)
    cbar.ax.set_yticks([])
    cbar.ax.set_yticklabels([])
    cbar.set_label('Density of KMC supersteps')
    

    plt.savefig("timeline.jpg", dpi=300)

# Specify the path to your output.txt file
file_path = 'output.txt'

# Extract time points and plot the timeline
time_points = extract_time_points(file_path)
plot_timeline(time_points)
