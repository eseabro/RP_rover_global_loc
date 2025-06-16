import numpy as np
import cv2

class GlobalBEVBuilder:
    def __init__(self, 
                 x_range=(-100, 100), 
                 y_range=(-100, 100),
                 resolution=0.1,
                 height_range=(-5, 5)):
        """
        Initialize a Global BEV map.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.height_range = height_range

        self.x_bins = int((x_range[1] - x_range[0]) / resolution)
        self.y_bins = int((y_range[1] - y_range[0]) / resolution)

        self.global_bev_map = np.zeros((self.y_bins, self.x_bins), dtype=np.uint8)

    def world_to_bev(self, points_world):
        """
        Convert world frame points to BEV map indices.
        """
        x_idx = ((points_world[:, 0] - self.x_range[0]) / self.resolution).astype(np.int32)
        y_idx = ((points_world[:, 1] - self.y_range[0]) / self.resolution).astype(np.int32)
        return x_idx, y_idx

    def add_points(self, points_world):
        """
        Add 3D points in world frame to the global BEV map.
        """
        # Filter points within the BEV range
        mask = (
            (points_world[:, 0] >= self.x_range[0]) & (points_world[:, 0] <= self.x_range[1]) &
            (points_world[:, 1] >= self.y_range[0]) & (points_world[:, 1] <= self.y_range[1]) &
            (points_world[:, 2] >= self.height_range[0]) & (points_world[:, 2] <= self.height_range[1])
        )
        points = points_world[mask]

        # Get BEV grid indices
        x_idx, y_idx = self.world_to_bev(points)

        # Safety: Clip indices
        x_idx = np.clip(x_idx, 0, self.x_bins - 1)
        y_idx = np.clip(y_idx, 0, self.y_bins - 1)

        # Mark as occupied
        self.global_bev_map[y_idx, x_idx] = 255

    def get_map(self):
        return self.global_bev_map


def build_local_bev_map(points_3d, x_range=(-20, 20), y_range=(-20, 20), resolution=0.2):
    """
    Create a BEV map (2D occupancy) from a 3D point cloud.
    """
    bev_map = np.zeros((int((y_range[1] - y_range[0]) / resolution),
                        int((x_range[1] - x_range[0]) / resolution)),
                       dtype=np.uint8)

    for p in points_3d:
        x, y, z = p
        if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
            ix = int((x - x_range[0]) / resolution)
            iy = int((y - y_range[0]) / resolution)
            bev_map[iy, ix] = 255  # mark occupied

    return bev_map


def extract_bev_features(bev_image):
    orb = cv2.ORB_create(1000)
    keypoints, descriptors = orb.detectAndCompute(bev_image, None)
    return keypoints, descriptors


