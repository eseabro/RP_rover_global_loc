from sklearn.linear_model import RANSACRegressor
import numpy as np
import cv2

class RANSACMatcher:
    def __init__(self, threshold=5.0, max_trials=1000):
        self.threshold = threshold
        self.max_trials = max_trials

    def match(self, source_points, target_points):
        # Ensure the points are in the correct shape
        source_points = np.array(source_points)
        target_points = np.array(target_points)

        if source_points.shape[1] != 2 or target_points.shape[1] != 2:
            raise ValueError("Points must be in shape (N, 2)")

        # Fit RANSAC model
        # ransac = RANSACRegressor(residual_threshold=self.threshold, max_trials=self.max_trials)
        # ransac.fit(source_points, target_points)
        H, inlier_mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, self.threshold)

        # # Get inliers and outliers
        # inliers = ransac.inlier_mask_
        # outliers = np.logical_not(inliers)

        # return inliers, outliers

        if inlier_mask is None:
            inlier_mask = np.zeros((len(source_points),), dtype=bool)
        else:
            inlier_mask = inlier_mask.ravel().astype(bool)

        outlier_mask = ~inlier_mask
        return inlier_mask, outlier_mask
    
    def set_threshold(self, threshold):
        self.threshold = threshold
        print(f"Threshold set to: {self.threshold}")

