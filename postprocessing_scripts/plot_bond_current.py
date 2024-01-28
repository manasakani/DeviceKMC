import numpy as np
import plotly.graph_objects as go

# Load xyz coordinates from a file (skip the first two rows and ignore the first entry in each line)
xyz_file = 'snapshot_0.xyz'
xyz_data = np.loadtxt(xyz_file, skiprows=2, usecols=(1, 2, 3))  # Skip the first two rows and use columns 2, 3, and 4

# Load current flow matrix from the C++ output file
X_file = 'X.txt'
X = np.loadtxt(X_file)

# Calculate the magnitude of the outgoing currents at each coordinate point
magnitude = np.linalg.norm(np.log(-X), axis=1)  # Axis 1 corresponds to the outgoing currents

# Create a 3D scatter plot for nodes
scatter = go.Scatter3d(
    x=xyz_data[:, 0],
    y=xyz_data[:, 1],
    z=xyz_data[:, 2],
    mode='markers',
    marker=dict(size=4, color='red')
)

# Create a quiver plot for vector field
quiver = go.Cone(
    x=xyz_data[:, 0],
    y=xyz_data[:, 1],
    z=xyz_data[:, 2],
    u=X[:, 0],
    v=X[:, 1],
    w=X[:, 2],
    sizemode="absolute",
    sizeref=0.01,
    colorbar=dict(thickness=20, ticklen=4)
)

# Create an isosurface for the magnitude
isosurface = go.Isosurface(
    x=xyz_data[:, 0],
    y=xyz_data[:, 1],
    z=xyz_data[:, 2],
    value=magnitude,
    isomin=np.min(magnitude),
    isomax=np.max(magnitude) * 0.7,  # Adjust the threshold here
    surface_count=8,
    colorbar=dict(thickness=20, ticklen=4)
)

# Create a layout
layout = go.Layout(
    scene=dict(aspectmode="data"),
    showlegend=False
)

# Combine scatter, quiver, and isosurface plots
fig = go.Figure(data=[scatter, quiver, isosurface], layout=layout)

# Show the plot
fig.write_image('output_figure.jpg')

