import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def show_icp_overlay(global_img, local_img, R, t, alpha=0.5):
    """
    Show only the white lines in local_img as blue over the global grayscale map.
    """
    h_g, w_g = global_img.shape
    h_l, w_l = local_img.shape

    # Build 2x3 affine transform matrix
    M = np.hstack([R, t.reshape(2, 1)])

    # Convert global image to color
    global_color = cv2.cvtColor(global_img, cv2.COLOR_GRAY2BGR)

    # Threshold local image to get white lines (you can tune the threshold)
    _, mask = cv2.threshold(local_img, 200, 255, cv2.THRESH_BINARY)

    # Create blue overlay (BGR)
    local_blue = np.zeros((h_l, w_l, 3), dtype=np.uint8)
    local_blue[mask > 0] = [255, 0, 0]  # Blue where white lines are

    # Warp blue overlay into global image space
    warped_blue = cv2.warpAffine(local_blue, M, (w_g, h_g))

    # Combine global and warped blue with transparency
    overlay = cv2.addWeighted(global_color, 1 - alpha, warped_blue, alpha, 0)

    # Show result
    plt.figure(figsize=(10, 8))
    plt.title("Blue Local Map Lines Overlaid on Global Map")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

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


def extract_largest_contour_points(binary_img, max_points=300):
    # ✅ Ensure the image is correct type
    binary_img = binary_img.astype(np.uint8)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("[Warning] No contours found.")
        return np.empty((0, 2), dtype=np.float32)

    # Find the largest contour by arc length
    largest = max(contours, key=cv2.contourArea)

    points = largest[:, 0, :]  # shape (N, 2)

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    return points.astype(np.float32)



def main():
    img1 = cv2.imread('src/img/global_map.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('src/img/local_map_2.png', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('src/img/local_map_crop.png', cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None or img3 is None:
        print("Failed to load one or more images.")
        return

    # Threshold both images to get binary contours
    bin1 = cv2.Canny(img1, 50, 150)
    bin2 = cv2.Canny(img2, 50, 150)
    bin3 = cv2.Canny(img3, 50, 150)


    pts1 = extract_largest_contour_points(bin1)
    pts2 = extract_largest_contour_points(bin2)
    pts3 = extract_largest_contour_points(bin3)

    if pts1 is None or pts2 is None or pts3 is None:
        print("Could not extract contours.")
        return

    ### ----- First alignment (img2 to img1) -----
    print(f"[ICP Input] Local (source) points: {pts2.shape}")
    print(f"[ICP Input] Global (target) points: {pts1.shape}")
    if pts1.shape != pts2.shape:
        min_len = min(pts1.shape[0], pts2.shape[0])
        pts1 = pts1[:min_len]
        pts2 = pts2[:min_len]
    H, _ = cv2.estimateAffinePartial2D(pts2, pts1)

    if H is None:
        print("Failed to estimate transform from local to global.")
        return

    print("[Affine] Initial H:\n", H)

    # Run ICP for refinement
    _, _, result = run_icp(pts2, pts1, H)

    # Extract and show transform
    T = result.transformation
    R = T[:2, :2]
    t = T[:2, 3]

    print("[ICP] Refined transform (img2 → img1):")
    show_icp_overlay(img1, img2, R, t)

    ### ----- Second alignment (img3 to img1) -----
    if pts1.shape != pts3.shape:
        min_len = min(pts1.shape[0], pts3.shape[0])
        pts1 = pts1[:min_len]
        pts3 = pts3[:min_len]
    H2, _ = cv2.estimateAffinePartial2D(pts3, pts1)
    if H2 is None:
        print("Failed to estimate transform from cropped room to global.")
        return

    _, _, result2 = run_icp(pts3, pts1, H2)
    T2 = result2.transformation
    R2 = T2[:2, :2]
    t2 = T2[:2, 3]

    print("[ICP] Refined transform (img3 → img1):")
    show_icp_overlay(img1, img3, R2, t2)

if __name__ == "__main__":
    main()