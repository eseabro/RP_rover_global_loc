import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


class RANSACTransformer:
    def __init__(self, points_local, points_global):
        '''
        points_local: (N, 2) array of observed points (from stereo camera)
        points_global: (N, 2) array of corresponding points in the global map
        '''
        # Add ones for translation term
        points_local_h = np.hstack([points_local, np.ones((points_local.shape[0], 1))])

        # Fit a linear model with RANSAC
        self.model = RANSACRegressor(LinearRegression(), residual_threshold=1.0, max_trials=1000)
        ## Alternative option: matrix, inliers = cv2.estimateAffinePartial2D(points_local, points_global, method=cv2.RANSAC, ransacReprojThreshold=1.0)

        self.model.fit(points_local_h, points_global)

        # Retrieve best transformation
        self.coef = self.model.estimator_.coef_  # (2, 3) matrix
        # coef will look like:
        # [[r11, r12, tx],
        #  [r21, r22, ty]]
        self.translation = self.model.estimator_.intercept_  # (2,) translation vector
    

    def transform(self, points_local):
        points_local_h = np.hstack([points_local, np.ones((points_local.shape[0], 1))])
        return self.model.predict(points_local_h)
    