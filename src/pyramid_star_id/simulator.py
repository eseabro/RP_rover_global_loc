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
    catalog, num_true=8, num_false=4, noise_deg=0.01, seed=0
):
    rng = np.random.RandomState(seed)
    n_catalog = len(catalog)
    if n_catalog == 0:
        raise ValueError("Catalog is empty")
    
    # --- 1. Generate random rover position within catalog region ---
    cat_xy = np.stack([[d['lon_deg'], d['lat_deg']] for d in catalog], axis=0)
    
    # Get catalog bounds
    lat_min, lat_max = cat_xy[:, 1].min(), cat_xy[:, 1].max()
    lon_min, lon_max = cat_xy[:, 0].min(), cat_xy[:, 0].max()
    
    # Add some margin (10% of region size) so rover can be near edges
    lat_margin = (lat_max - lat_min) * 0.1
    lon_margin = (lon_max - lon_min) * 0.1
    
    rover_lat = rng.uniform(lat_min + lat_margin, lat_max - lat_margin)
    rover_lon = rng.uniform(lon_min + lon_margin, lon_max - lon_margin)
    
    theta = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[c, -s], [s,  c]])
    t_rover = np.array([rover_lon, rover_lat])
    print(f"Rover position (lon, lat): {t_rover}, heading: {np.degrees(theta):.2f}°")
    
    # --- 2. Select closest catalog landmarks ---
    dists = np.linalg.norm(cat_xy - t_rover, axis=1)
    closest_idx = np.argsort(dists)[:min(num_true, n_catalog)]
    true_points = cat_xy[closest_idx]
    print(f"Selected true landmark indices: {closest_idx}")
    print(f"Max observation distance: {dists[closest_idx].max():.2f}°")
    
    # --- 3. Transform to rover local frame ---
    true_local = (true_points - t_rover) @ R2.T
    
    # --- 4. Generate false landmarks ---
    r_true = np.linalg.norm(true_local, axis=1)
    spread = np.mean(r_true) if len(r_true) > 0 else 1.0
    false_r = rng.uniform(0.5*spread, 1.5*spread, size=num_false)
    false_theta = rng.uniform(0, 2*np.pi, size=num_false)
    false_pts = np.column_stack([
        false_r * np.cos(false_theta),
        false_r * np.sin(false_theta)
    ])
    
    # --- 5. Add noise to true observations ---
    sigma_rad = np.deg2rad(noise_deg)
    observed_true = []
    for p in true_local:
        r = np.linalg.norm(p)
        angle = np.arctan2(p[1], p[0]) + rng.normal(0, sigma_rad)
        observed_true.append([r * np.cos(angle), r * np.sin(angle)])
    observed_true = np.array(observed_true)
    
    # --- Combine true + false ---
    lat_lon_vectors = np.vstack([observed_true, false_pts])
    
    # --- Convert to km using ref_lat=0.0 (equator) ---
    ref_lat = 0.0
    deg_to_km_lat = 111.0
    deg_to_km_lon = 111.0 * np.cos(np.deg2rad(ref_lat))  # = 111.0
    
    km_vectors = np.column_stack([
        lat_lon_vectors[:,0] * deg_to_km_lon,
        lat_lon_vectors[:,1] * deg_to_km_lat
    ])
    
    return {
        "observed_vectors": km_vectors,
        "lat_lon_vectors": lat_lon_vectors,
        "true_indices": list(closest_idx),
        "R_true": R2,
        "t_rover": t_rover,
        "n_false": num_false
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