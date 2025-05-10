import pyzed.sl as sl
import cv2
import numpy as np
from ROS_ws.src.slam_matcher.slam_matcher.bev_builder import GlobalBEVBuilder, build_local_bev_map, extract_bev_features

def init_zed():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    camera_info = zed.get_camera_information()
    left_cam = camera_info.calibration_parameters.left_cam

    camera_intrinsics = {
        'fx': left_cam.fx,
        'fy': left_cam.fy,
        'cx': left_cam.cx,
        'cy': left_cam.cy
    }

    return zed, camera_intrinsics

def grab_frame(zed):
    runtime_parameters = sl.RuntimeParameters()
    image_left = sl.Mat()
    depth_map = sl.Mat()

    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        frame = image_left.get_data()[:, :, :3]  # BGRA or BGR depending on SDK
        depth = depth_map.get_data()[:, :, 0]

        return frame, depth
    else:
        return None, None
    

def grab_zed_images(zed):
    """
    Grab left and right stereo images from a ZED camera.

    Args:
        zed: Initialized sl.Camera() object.

    Returns:
        left_img (np.ndarray): Left camera image (grayscale or color).
        right_img (np.ndarray): Right camera image (grayscale or color).
    """
    runtime_parameters = sl.RuntimeParameters()

    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        image_left = sl.Mat()
        image_right = sl.Mat()

        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        zed.retrieve_image(image_right, sl.VIEW.RIGHT)

        left_np = image_left.get_data()[:, :, :3]  # (H, W, 3) BGR
        right_np = image_right.get_data()[:, :, :3]  # (H, W, 3) BGR

        return left_np, right_np

    else:
        return None, None
    
def feed_slam(slam_system, left_img, right_img, timestamp):
    """
    Feed stereo images into ORB-SLAM3 system and get pose + map points.

    Args:
        slam_system: ORB-SLAM3 System instance.
        left_img (np.ndarray): Left camera image.
        right_img (np.ndarray): Right camera image.
        timestamp (float): Timestamp for the frame.

    Returns:
        pose (np.ndarray): 4x4 SE3 pose matrix (world to camera).
        map_points (list): List of map points (each a 3D coordinate).
    """
    # ORB-SLAM3 expects grayscale images
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 1. Track with ORB-SLAM3
    pose = slam_system.TrackStereo(left_gray, right_gray, timestamp)

    if pose is None:
        return None, []

    # 2. Get tracked map points
    map_points = slam_system.GetTrackedMapPoints()

    # Convert C++ point class to numpy (if necessary)
    points_3d = []
    for p in map_points:
        if p is not None:
            pos = p.GetWorldPos()  # p is a MapPoint object
            points_3d.append(np.array([pos[0], pos[1], pos[2]]))

    return pose, points_3d


def extract_features(image_gray, detector):
    keypoints, descriptors = detector.detectAndCompute(image_gray, None)
    return keypoints, descriptors

def match_features(descriptors_local, descriptors_global):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_local, descriptors_global)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def draw_transformed_points(frame, points):
    for p in points.astype(int):
        cv2.circle(frame, (p[0], p[1]), 2, (0, 255, 0), -1)
    return frame

def get_local_map_from_orbslam(pose, map_points, radius=30.0):
    """
    Select nearby 3D map points within a radius around the current pose.
    """
    t = pose[:3, 3]  # Current camera position
    local_points = []

    for p in map_points:
        if p is None:
            continue
        xyz = p.getWorldPos()  # Get 3D position (ORB-SLAM3 C++ method)
        dist = np.linalg.norm(xyz - t)
        if dist < radius:
            local_points.append(xyz)

    return np.array(local_points)


def main():
    # 1. Initialize
    zed, camera_intrinsics = init_zed()
    orb = cv2.ORB_create(2000)
    global_bev = GlobalBEVBuilder()

    # 2. Load or prepare global map (dummy example here)
    global_map_image = np.zeros((720, 1280), dtype=np.uint8)  # <-- Load your real map here
    keypoints_global, descriptors_global = extract_features(global_map_image, orb)

    while True:
        left_img, right_img = grab_zed_images(zed)
        if left_img is None or right_img is None:
            continue

        timestamp = sl.get_current_timestamp()  # Or your own time manager

        pose, map_points = feed_slam(slam_system, left_img, right_img, timestamp)

        if pose is None:
            continue

        points_3d_local = get_local_map_from_orbslam(pose, map_points)

        if len(points_3d_local) == 0:
            continue

        # 1. Build local BEV map
        local_bev_map = build_local_bev_map(points_3d_local)

        # 2. Extract features from local BEV
        keypoints_local, descriptors_local = extract_bev_features(local_bev_map)

        if descriptors_local is None or descriptors_global is None:
            continue  # Skip if no features

        matches = match_features(descriptors_local, descriptors_global)

        if len(matches) < 4:
            continue  # Need at least 4 matches for RANSAC

        matched_points_local = np.array([keypoints_local[m.queryIdx].pt for m in matches], dtype=np.float32)
        matched_points_global = np.array([keypoints_global[m.trainIdx].pt for m in matches], dtype=np.float32)

        matrix, inliers = cv2.estimateAffinePartial2D(
            matched_points_local,
            matched_points_global,
            method=cv2.RANSAC,
            ransacReprojThreshold=1.0
        )

        if matrix is not None:
            points_local = np.array([kp.pt for kp in keypoints_local], dtype=np.float32)
            transformed_points = cv2.transform(np.expand_dims(points_local, axis=0), matrix)[0]
            frame_out = draw_transformed_points(frame.copy(), transformed_points)

            cv2.imshow("Aligned Frame", frame_out)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
