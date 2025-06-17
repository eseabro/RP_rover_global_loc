import cv2
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def extract_features(img1, img2, img3=None):
    sift = cv2.BRISK_create()
    #     threshold=0.0003,         # lower threshold = more detections
    #     nOctaves=5,
    #     nOctaveLayers=6,
    #     descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
    #     descriptor_channels=3
    # )

    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    img_with_kp = cv2.drawKeypoints(img1, keypoints1, None)
    plt.imshow(img_with_kp)
    plt.show()
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    img_with_kp2 = cv2.drawKeypoints(img2, keypoints2, None)
    plt.imshow(img_with_kp2)
    plt.show()
    print(f"[SIFT] Extracted {len(keypoints1)} keypoints from img1")
    print(f"[SIFT] Extracted {len(keypoints2)} keypoints from img2")

    if img3 is not None:
        keypoints3, descriptors3 = sift.detectAndCompute(img3, None)
        img_with_kp3 = cv2.drawKeypoints(img3, keypoints3, None)
        plt.imshow(img_with_kp3)
        plt.show()
        print(f"[SIFT] Extracted {len(keypoints3)} keypoints from img3")

    else:
        keypoints3, descriptors3 = None, None
    return keypoints1, descriptors1, keypoints2, descriptors2, keypoints3, descriptors3

def extract_LSD_features(img, method="endpoints"):

    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(img)[0]

    keypoints = []

    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = line[0]
            if method == "endpoints":
                keypoints.append(cv2.KeyPoint(x0, y0, 1))
                keypoints.append(cv2.KeyPoint(x1, y1, 1))
            elif method == "midpoints":
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                keypoints.append(cv2.KeyPoint(mx, my, 1))
    print(f"[LSD] Extracted {len(keypoints)} keypoints using {method} method")
    return keypoints


def ransac_filter(pts1, pts2):
    H, mask = cv2.estimateAffine2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=15.0)
    inliers = mask.ravel().tolist()
    print(f"[RANSAC] Found {np.sum(inliers)} inliers out of {len(inliers)} matches")
    return H, inliers

def match_knn_features(desc1, desc2, ratio=0.9):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(desc1, desc2)
    good_matches = [m for m, n in knn_matches if m.distance < ratio * n.distance]
    print(f"[Match] Found {len(knn_matches)} raw knn matches")
    print(f"[Match] Kept {len(good_matches)} matches after Lowe's ratio test")
    print("Mean match distance:", np.mean([m.distance for m in good_matches]))
    return sorted(good_matches, key=lambda x: x.distance)

def match_features(desc1, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"[Match] Found {len(matches)} total matches")
    print("Mean match distance:", np.mean([m.distance for m in matches]))
    return matches

def match_by_proximity(kp1, kp2, max_distance=20):
    pts1 = np.array([kp.pt for kp in kp1])
    pts2 = np.array([kp.pt for kp in kp2])
    
    matches = []
    for i, p1 in enumerate(pts1):
        dists = np.linalg.norm(pts2 - p1, axis=1)
        j = np.argmin(dists)
        if dists[j] < max_distance:
            matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=dists[j]))
    
    return matches

def get_matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    return pts1, pts2

def to_open3d_cloud(points):
    pcd = o3d.geometry.PointCloud()
    points_3d = np.hstack((points, np.zeros((points.shape[0], 1))))  # z = 0
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    return pcd

def run_icp(source_points, target_points, H, threshold=30.0):

    print(f"[ICP Input] Local (source) points: {source_points.shape}")
    print(f"[ICP Input] Global (target) points: {target_points.shape}")

    source = to_open3d_cloud(source_points)
    target = to_open3d_cloud(target_points)

    t_init = np.eye(4)
    t_init[:2, :2] = H[:, :2]
    t_init[:2, 3] = H[:, 2]


    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, t_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    print("ICP Transformation Matrix:\n", result.transformation)
    print(f"[ICP] Fitness: {result.fitness:.4f}")
    print(f"[ICP] Inlier RMSE: {result.inlier_rmse:.4f}")


    return source, target, result

def show_keypoint_matches(img1, kp1, img2, kp2, matches, inlier_mask=None, title="Matches"):
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches,
        None,
        matchesMask=inlier_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    # Show result
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.imshow(img_matches)
    plt.axis('off')
    plt.show()

def show_icp_overlay(global_img, local_img, R, t, alpha=0.5):
    """
    Overlay the entire local_img as a bluescale transparent layer over global_img using a 2D transform.
    """
    h_g, w_g = global_img.shape
    h_l, w_l = local_img.shape

    # Build 2x3 affine transform matrix
    M = np.hstack([R, t.reshape(2, 1)])

    # Convert global image to BGR color
    global_color = cv2.cvtColor(global_img, cv2.COLOR_GRAY2BGR)

    # Convert local image to "bluescale" (only blue channel gets intensity)
    local_blue = np.zeros((h_l, w_l, 3), dtype=np.uint8)
    local_blue[:, :, 0] = local_img  # Blue channel gets intensity

    # Warp the blue-tinted local image into the global image space
    warped_blue = cv2.warpAffine(local_blue, M, (w_g, h_g), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Blend global image and blue overlay
    overlay = cv2.addWeighted(global_color, 1 - alpha, warped_blue, alpha, 0)

    # Show result
    plt.figure(figsize=(10, 8))
    plt.title("Bluescale Local Image Overlaid on Global Map")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()



def estimate_crop_offset(global_pts, local_pts):
    return np.median(global_pts - local_pts, axis=0)

def main():
    img1 = cv2.imread('src/custom_slam/images/global_map.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('src/custom_slam/images/local_control.png', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('src/custom_slam/images/test2.png', cv2.IMREAD_GRAYSCALE)



    if img1 is None or img2 is None:
        print("Failed to load one or both images.")
        return
    
    # img1 = (skeletonize(img1) * 255).astype(np.uint8)
    # img2 = (skeletonize(img2) * 255).astype(np.uint8)
    # img3 = (skeletonize(img3) * 255).astype(np.uint8)
    # img2_cleaned = img2


    kp1, desc1, kp2, desc2, kp3, desc3 = extract_features(img1, img2, img3)



    matches = match_features(desc1, desc2)
    

    pts1, pts2 = get_matched_points(kp1, kp2, matches)

    H, inliers = ransac_filter(pts1, pts2)

    if H is None:
        print("RANSAC failed to find a valid affine transform.")
        return

    show_keypoint_matches(img1, kp1, img2, kp2, matches, inliers)

    print("RANSAC Homography Matrix:\n", H)
    inlier_mask = np.array(inliers, dtype=bool)
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]

    print("Running ICP on inliers...")
    source_cloud, target_cloud, result = run_icp(pts2_inliers, pts1_inliers, H)



    T = result.transformation  # 4x4 matrix
    R = T[:2, :2]              # 2x2 rotation matrix
    t = T[:2, 3]               # 2D translation vector

    print("ICP Result:")
    # ✅ Overlay aligned local map using corrected transform
    show_icp_overlay(img1, img2, R, t)

    if kp3 is not None and desc3 is not None:
        matches2 = match_features(desc1, desc3)
        pts1, pts2 = get_matched_points(kp1, kp3, matches2)

        H, inliers = ransac_filter(pts1, pts2)

        if H is None:
            print("RANSAC failed to find a valid affine transform.")
            return


        show_keypoint_matches(img1, kp1, img3, kp3, matches2, inliers)
        inlier_mask = np.array(inliers, dtype=bool)
        pts1_inliers = pts1[inlier_mask]
        pts2_inliers = pts2[inlier_mask]

        source_cloud, target_cloud, result = run_icp(pts2_inliers, pts1_inliers, H)



        T = result.transformation  # 4x4 matrix
        R = T[:2, :2]              # 2x2 rotation matrix
        t = T[:2, 3]               # 2D translation vector


        # ✅ Overlay aligned local map using corrected transform
        show_icp_overlay(img1, img3, R, t)

    # Visualization
    # source_transformed = copy.deepcopy(source_cloud).transform(result.transformation)
    # source_cloud.paint_uniform_color([1, 0, 0])           # Red = original source
    # target_cloud.paint_uniform_color([0, 1, 0])           # Green = target
    # source_transformed.paint_uniform_color([0, 0, 1])     # Blue = aligned source

    # plot_icp_alignment(target_cloud, source_cloud, source_transformed)
    # o3d.visualization.draw_geometries([target_cloud, source_cloud, source_transformed],
    #                                   window_name="ICP Alignment")


if __name__ == "__main__":
    main()
