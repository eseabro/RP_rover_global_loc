import numpy as np
import matplotlib.pyplot as plt

def create_bev(points_3d, 
               x_range=(-50, 50), 
               y_range=(-50, 50), 
               resolution=0.1,
               height_range=(-5, 5)):
    """
    Creates a BEV (Bird's Eye View) map from 3D points.
    
    Args:
        points_3d (np.ndarray): (N, 3) array of (x, y, z) points.
        x_range (tuple): Min and max values for X axis (meters).
        y_range (tuple): Min and max values for Y axis (meters).
        resolution (float): Size of each cell in meters.
        height_range (tuple): Min and max Z heights for filtering.
    
    Returns:
        bev_map (np.ndarray): (H, W, 2) BEV map with occupancy and height channels.
    """

    # 1. Filter points within the specified 3D bounding box
    mask = (
        (points_3d[:, 0] >= x_range[0]) & (points_3d[:, 0] <= x_range[1]) &
        (points_3d[:, 1] >= y_range[0]) & (points_3d[:, 1] <= y_range[1]) &
        (points_3d[:, 2] >= height_range[0]) & (points_3d[:, 2] <= height_range[1])
    )
    points = points_3d[mask]

    # 2. Define the size of the BEV map
    x_bins = int((x_range[1] - x_range[0]) / resolution)
    y_bins = int((y_range[1] - y_range[0]) / resolution)

    bev_map = np.zeros((y_bins, x_bins, 2), dtype=np.float32)  # channels: occupancy, height

    # 3. Convert 3D coordinates to BEV map indices
    x_indices = ((points[:, 0] - x_range[0]) / resolution).astype(np.int32)
    y_indices = ((points[:, 1] - y_range[0]) / resolution).astype(np.int32)

    # 4. Fill occupancy and height maps
    for x_idx, y_idx, z in zip(x_indices, y_indices, points[:, 2]):
        if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
            bev_map[y_idx, x_idx, 0] = 1  # Occupancy
            bev_map[y_idx, x_idx, 1] = max(bev_map[y_idx, x_idx, 1], z)  # Max height

    return bev_map

def visualize_bev(bev_map):
    """
    Visualizes BEV map (Occupancy + Height)
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(bev_map[:, :, 0], cmap='gray')
    ax[0].set_title('Occupancy Map')
    
    ax[1].imshow(bev_map[:, :, 1], cmap='plasma')
    ax[1].set_title('Height Map')
    
    plt.show()

# ==== Example Usage ====

# Example: generate random 3D points (you would use your SLAM + stereo output)
np.random.seed(42)
points_3d = np.random.uniform(low=[-60, -60, -5], high=[60, 60, 5], size=(10000, 3))

# Create BEV map
bev_map = create_bev(points_3d)

# Visualize it
visualize_bev(bev_map)
