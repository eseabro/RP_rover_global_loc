"""Observation simulator: rotate subset of catalog stars to body frame, add noise and false detections."""
import numpy as np
from .geometry import apply_rotation, random_rotation_matrix
from .catalog import lat_lon_to_vector

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

def normalize_rows(a):
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n

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


def sample_observed(catalog, n_obs=10, seed=1, add_translation=True, pos_noise_m=0.5, ang_noise_deg=0.2):
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(catalog), size=min(n_obs, len(catalog)), replace=False)
    axis = normalize_rows(rng.randn(1,3))[0]
    angle = (rng.rand() - 0.5) * 2 * np.pi
    K = np.array([[0, -axis[2], axis[1]],[axis[2], 0, -axis[0]],[-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
    t = rng.randn(3) * (50.0 if add_translation else 0.0)
    observed = []
    for idx in chosen:
        e = catalog[idx]
        pos = np.array([e['pos_x'], e['pos_y'], e['pos_z']])
        vec = np.array([e['vec_x'], e['vec_y'], e['vec_z']])
        pos_r = R @ pos + t
        vec_r = R @ vec
        pos_r = pos_r + rng.randn(3) * pos_noise_m
        ang = np.radians(ang_noise_deg) * rng.randn()
        ax = normalize_rows(rng.randn(1,3))[0]
        K2 = np.array([[0, -ax[2], ax[1]],[ax[2], 0, -ax[0]],[-ax[1], ax[0], 0]])
        Rn = np.eye(3) + np.sin(ang)*K2 + (1-np.cos(ang))*(K2@K2)
        vec_r = Rn @ vec_r
        vec_r = vec_r / np.linalg.norm(vec_r)
        observed.append({
            'id': e['id'],
            'pos': pos_r,
            'vec': vec_r,
            'type': e['type'],
            'radius': e['radius']
        })
    ground_truth = {'R': R, 't': t, 'selected_ids': list(chosen)}
    return observed, ground_truth

import numpy as np

def simulate_observations_with_pose(
    catalog, num_true=8, num_false=4, noise_deg=0.01, fov_deg=180, seed=0
):
    rng = np.random.RandomState(seed)
    n = len(catalog)

    # --- 1. Pick which catalog features are visible ---
    true_indices = rng.choice(range(n), size=min(num_true, n), replace=False)
    true_planet = np.stack([[catalog[i]['x'], catalog[i]['y']] for i in true_indices], axis=0)

    # --- 2. Random rover position in same 2D coordinate system ---
    rover_lat = rng.uniform(-70, 70)
    rover_lon = rng.uniform(-160, 160)
    print(f"Rover lat, lon: {rover_lat:.3f}, {rover_lon:.3f}")
    t_rover = np.array([rover_lon, rover_lat])   # shape (2,)

    # --- 3. Transform landmarks into rover local frame ---
    points_rel = true_planet - t_rover  # rover at (0,0)

    # --- 4. Random 2D rotation (camera yaw in the plane) ---
    theta = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[c, -s],
                   [s,  c]])  # 2x2
    print("True rotation matrix:\n", R2)

    true_cam = points_rel @ R2.T  # rotate into camera frame

    # --- 5. Field of view filtering (simple angular filter in 2D) ---
    # Here we approximate FOV by angle from +X axis
    angles = np.degrees(np.arctan2(true_cam[:,1], true_cam[:,0]))
    in_fov = np.abs(angles) <= fov_deg/2
    true_cam = true_cam[in_fov]
    true_indices = np.array(true_indices)[in_fov]

    # --- 6. Add angular noise (small rotation perturbation) ---
    sigma_rad = np.deg2rad(noise_deg)
    observed_true = []
    for v in true_cam:
        angle = np.arctan2(v[1], v[0])
        angle += rng.normal(0, sigma_rad)
        r = np.linalg.norm(v)
        observed_true.append([r*np.cos(angle), r*np.sin(angle)])
    observed_true = np.array(observed_true)

    # --- 7. Simulate false detections randomly in FOV ---
    num_false = int(num_false)
    false_angles = rng.uniform(-np.deg2rad(fov_deg/2), np.deg2rad(fov_deg/2), num_false)
    false_r = rng.uniform(0.5, 1.5, num_false)  # random radius
    fx = false_r * np.cos(false_angles)
    fy = false_r * np.sin(false_angles)
    false_pts = np.stack([fx, fy], axis=1)

    # --- 8. Combine true + false ---
    observed_all = np.vstack([observed_true, false_pts])

    return {
        "observed_vectors": observed_all,   # Nx2 points in camera frame
        "true_indices": list(true_indices),
        "R_true": R2,                       # 2D rotation matrix
        "t_rover": t_rover                  # rover global position in 2D
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