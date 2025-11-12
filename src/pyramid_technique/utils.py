import numpy as np
from math import radians, degrees
from collections import defaultdict
np.random.seed(42)
# -------------------------
# Utility functions
# -------------------------
def ra_dec_to_vector(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    v = np.stack([x, y, z], axis=-1)
    # normalize (should already be unit)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return v

def angular_distance_deg(v1, v2):
    # return angular distance in degrees between unit vectors v1 and v2
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def kabsch_rotation(A, B):
    # A, B: Nx3 arrays of corresponding points (unit vectors)
    # Finds rotation R such that R @ A.T ~ B.T
    # Use SVD-based Kabsch (no scaling)
    assert A.shape == B.shape
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R