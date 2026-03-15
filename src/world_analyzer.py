import trimesh
import imageio.v3 as iio
import numpy as np

# Load your 2020 Marsyard Mesh
scene = trimesh.load('ROS_ws/src/custom_slam/models/cnes_marsyard/meshes/serom.obj')
img = iio.imread('ROS_ws/src/leo_simulator-ros2/leo_gz_worlds/models/marsyard2022_terrain/dem/marsyard2022_terrain_hm.tif')

print("=== Marsyard CNES Terrain Analysis ===")

# 2. Force concatenate every piece of geometry in the scene into ONE mesh
if isinstance(scene, trimesh.Scene):
    mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
else:
    mesh = scene

# 3. Apply the 90-degree X-axis rotation from Gazebo
rot_matrix = trimesh.transformations.rotation_matrix(1.5708, [1, 0, 0])
mesh.apply_transform(rot_matrix)

# 3. Calculate True Altitude Difference and Size
min_bounds, max_bounds = mesh.bounds
size_x = max_bounds[0] - min_bounds[0]
size_y = max_bounds[1] - min_bounds[1]
size_z = max_bounds[2] - min_bounds[2]

print(f"Dimensions: X={size_x:.2f}m, Y={size_y:.2f}m")
print(f"Max Altitude Difference (Z): {size_z:.2f}m")

# 4. Calculate Maximum Slope
z_axis = np.array([0.0, 0.0, 1.0])
cos_angles = np.dot(mesh.face_normals, z_axis)

angles_rad = np.arccos(np.clip(cos_angles, -1.0, 1.0))
angles_deg = np.degrees(angles_rad)

# Ignore straight 90-degree drop-offs (usually the outer bounding box walls)
actual_slopes = angles_deg[angles_deg < 89.0] 

# Use the 99.9th percentile to ignore random single-triangle spikes
steepest_driveable = np.percentile(actual_slopes, 99.9)

print(f"Max Driveable Slope: {steepest_driveable:.2f} degrees")
print(f"Average Slope: {np.mean(actual_slopes):.2f} degrees")


print("=== Marsyard 2022 Terrain Analysis ===")

# 1. Load the Heightmap image

# Normalize the image to range between 0.0 and 1.0
img_normalized = (img - img.min()) / (img.max() - img.min())

# Apply the Gazebo Z-scale from your XML file
z_scale = 4.820803273566
elevations = img_normalized * z_scale

# Calculate spatial resolution (meters per pixel)
# The Marsyard 2022 XML says the map is 50x50 meters. 
pixels_x, pixels_y = img.shape
res_x = 50.0 / pixels_x
res_y = 50.0 / pixels_y

# 3. Calculate Gradients (the change in height between neighboring pixels)
dz_dy, dz_dx = np.gradient(elevations, res_y, res_x)

# 4. Calculate Slope Math
# Slope = arctan(sqrt( (dz/dx)^2 + (dz/dy)^2 ))
slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
slope_deg = np.degrees(slope_rad)

print(f"Max Altitude Difference: {z_scale:.2f}m")
print(f"Max Slope: {np.max(slope_deg):.2f} degrees")
print(f"Average Slope: {np.mean(slope_deg):.2f} degrees")