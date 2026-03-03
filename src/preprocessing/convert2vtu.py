import numpy as np
import meshio

# Load data
points = np.loadtxt('endo_network_xyz.txt')           # Nx3
lines = np.loadtxt('endo_network_lines.txt', dtype=int)    # Mx2

# Convert from 1-based to 0-based indexing (MATLAB-style to Python-style)
if lines.min() == 1:
    lines -= 1

# Ensure points are 3D
if points.shape[1] == 2:
    points = np.hstack([points, np.zeros((points.shape[0], 1))])

# Create meshio object
mesh = meshio.Mesh(
    points=points,
    cells=[("line", lines)]
)

# Write as ASCII VTU (human-readable and ParaView-friendly)
meshio.write("line_mesh.vtu", mesh, file_format="vtu", binary=False)

print("✅ Successfully wrote 'line_mesh.vtu' (ASCII format). Open in ParaView.")
