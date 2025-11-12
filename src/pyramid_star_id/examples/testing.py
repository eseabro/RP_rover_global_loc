"""Demo script showing how to use the package."""
from pyramid_star_id import build_catalog
from pyramid_star_id.geometry import apply_rotation, kabsch_rotation
import numpy as np

def main():
    print('Building catalog...')
    catalog = build_catalog(n=60, seed=1)

    cat_quad = np.stack([catalog[i]['vec'] for i in [0,1,2,3]], axis=0)

    # Apply a known rotation
    theta = np.radians(45)
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    obs_quad = apply_rotation(R_true, cat_quad)
    R_est = kabsch_rotation(cat_quad, obs_quad)
    cat_rot = apply_rotation(R_est, cat_quad)
    residuals_deg = np.degrees(np.arccos(np.clip(np.sum(cat_rot * obs_quad, axis=1), -1, 1)))
    print(residuals_deg)
    print("R_true:\n", R_true)
    print("R_est:\n", R_est)
    print("Difference:\n", R_true - R_est)


if __name__ == '__main__':
    main()
