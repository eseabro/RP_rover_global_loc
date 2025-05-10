import numpy as np


def transform_points(points_cam, R_wc, t_wc):
    """
    Transform points from camera frame to world frame.

    Args:
        points_cam: (N, 3) array in camera frame
        R_wc: (3, 3) rotation matrix (world ← camera)
        t_wc: (3,) translation vector

    Returns:
        points_world: (N, 3) array in world frame
    """
    return (R_wc @ points_cam.T).T + t_wc


def depth_fuse_keypoints(keypoints, depth_map, camera_intrinsics):
    """
    Projects 2D keypoints with depth into 3D points in the camera frame.

    Args:
        keypoints: list of cv2.KeyPoint
        depth_map: (H, W) numpy array, depth in meters
        camera_intrinsics: dict with fx, fy, cx, cy

    Returns:
        points_3d_cam: (N, 3) array of 3D points in camera frame
    """
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    points_3d = []

    for kp in keypoints:
        u, v = int(kp.pt[0]), int(kp.pt[1])

        # Ignore keypoints outside depth map size
        if u < 0 or u >= depth_map.shape[1] or v < 0 or v >= depth_map.shape[0]:
            continue

        z = depth_map[v, u]

        # Ignore invalid depth
        if z <= 0 or np.isnan(z):
            continue

        # Backproject to 3D
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points_3d.append([x, y, z])

    if len(points_3d) == 0:
        return np.zeros((0, 3))

    return np.array(points_3d)
