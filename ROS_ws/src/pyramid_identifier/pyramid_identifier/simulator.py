"""Observation simulator: rotate subset of catalog stars to body frame, add noise and false detections."""
import numpy as np
from .geometry import apply_rotation, random_rotation_matrix

def perturb_unit_vector(v, sigma_rad, rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.normal(scale=sigma_rad)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
    v2 = R @ v
    return v2 / np.linalg.norm(v2)

def simulate_observations(catalog, num_true=8, num_false=4, noise_deg=0.01, fov_deg=180, seed=0):
    rng = np.random.RandomState(seed)
    n = len(catalog)

    # Pick which catalog features are visible
    true_indices = rng.choice(range(n), size=min(num_true, n), replace=False)
    true_planet = np.stack([catalog[i]['vec'] for i in true_indices], axis=0)

    # Random rover attitude (planet-fixed → camera)
    R_true = random_rotation_matrix(seed)
    print("True rotation matrix:\n", R_true)
    true_cam = apply_rotation(R_true, true_planet)

    # Only keep those within the camera field of view
    cos_fov = np.cos(np.deg2rad(fov_deg / 2))
    in_fov = true_cam[:, 2] > cos_fov
    true_cam = true_cam[in_fov]
    true_indices = np.array(true_indices)[in_fov]

    # Apply angular noise
    sigma_rad = np.deg2rad(noise_deg)
    observed_true = np.stack(
        [perturb_unit_vector(v, sigma_rad, rng) for v in true_cam], axis=0
    )

    # Simulate false detections randomly within same FOV
    num_false = int(num_false)
    az = rng.uniform(-np.deg2rad(fov_deg/2), np.deg2rad(fov_deg/2), num_false)
    el = rng.uniform(-np.deg2rad(fov_deg/2), np.deg2rad(fov_deg/2), num_false)
    fx = np.cos(el) * np.cos(az)
    fy = np.cos(el) * np.sin(az)
    fz = np.sin(el)
    false_vecs = np.stack([fx, fy, fz], axis=1)

    # Combine
    observed_all = np.vstack([observed_true, false_vecs])

    return {
        "observed_vectors": observed_all,
        "true_indices": list(true_indices),
        "R_true": R_true
    }

def simulate_identity_observations(catalog, num_true=8, seed=0):
    rng = np.random.RandomState(seed)
    n = len(catalog)

    # Randomly select a subset of catalog features
    true_indices = rng.choice(range(n), size=min(num_true, n), replace=False)

    # Directly take their vectors (no modification)
    observed_vectors = np.stack([catalog[i]['vec'] for i in true_indices], axis=0)

    # Identity rotation (since no transform was applied)
    R_true = np.eye(3)

    return {
        "observed_vectors": observed_vectors,
        "true_indices": list(true_indices),
        "R_true": R_true
    }