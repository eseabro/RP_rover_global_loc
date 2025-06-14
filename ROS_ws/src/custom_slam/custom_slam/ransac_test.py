import cv2
import numpy as np
from ransac_matcher import RANSACMatcher


def extract_features(img):
    """Extract keypoints using ORB (or any other detector)."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """Match descriptors using brute-force Hamming distance."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def get_matched_points(kp1, kp2, matches):
    """Convert matched keypoints to (x, y) coordinate arrays."""
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    tgt_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    return src_pts, tgt_pts

def main():
    # Load two grayscale images
    img1 = cv2.imread('src/img/global_map.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('src/img/local_map_2.png', cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error loading images.")
        return

    # Extract features and match them
    kp1, desc1 = extract_features(img1)
    kp2, desc2 = extract_features(img2)
    matches = match_features(desc1, desc2)
    src_pts, tgt_pts = get_matched_points(kp1, kp2, matches)

    # Run RANSAC matching
    matcher = RANSACMatcher(threshold=2.0, max_trials=1000)
    inliers, outliers = matcher.match(src_pts, tgt_pts)

    print(f"Inliers: {np.sum(inliers)} / {len(matches)}")

    # Optional: visualize matches
    inlier_matches = [m for i, m in enumerate(matches) if inliers[i]]
    img_match = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Inlier Matches", img_match)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
