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

def simulate_observations_with_pose(
    catalog,
    num_true=8,
    num_false=4,
    noise_deg=0.01,
    fov_deg=90,
    seed=0,
    mars_radius_km=3390.0
):
    """
    Simulate rover observations of Mars landmarks from a random surface position.

    The rover is placed randomly on the Mars surface (using lat/lon/elev).
    Observations are finite 3D landmarks -> camera direction vectors with noise.
    """
    rng = np.random.RandomState(seed)
    n = len(catalog)

    # --- 1. Pick a random rover location on Mars surface ---
    rover_lat = rng.uniform(-70, 70)        # avoid poles for stability
    rover_lon = rng.uniform(-180, 180)
    rover_elev = 0.0                        # can randomize small offsets if desired

    lat_rad = np.deg2rad(rover_lat)
    lon_rad = np.deg2rad(rover_lon)
    r = mars_radius_km + rover_elev

    # Mars-fixed Cartesian position of rover
    t_true = np.array([
        r * np.cos(lat_rad) * np.cos(lon_rad),
        r * np.cos(lat_rad) * np.sin(lon_rad),
        r * np.sin(lat_rad)
    ])

    # --- 2. Define a local "up" vector (surface normal) ---
    up = t_true / np.linalg.norm(t_true)

    # --- 3. Random camera orientation, but with camera Z-axis roughly aligned with -up (looking outward) ---
    # Create local tangent frame
    east = np.cross(np.array([0, 0, 1]), up)
    if np.linalg.norm(east) < 1e-6:  # near pole
        east = np.array([1, 0, 0])
    east /= np.linalg.norm(east)
    north = np.cross(up, east)

    # Base rotation from Mars-fixed frame → local frame
    R_surface = np.stack([east, north, up], axis=1)

    # Add a random yaw/pitch/roll perturbation
    yaw, pitch, roll = rng.uniform(-np.pi, np.pi, 3)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    # Final rover attitude (Mars-fixed → camera)
    R_true = R_roll @ R_pitch @ R_yaw @ R_surface.T

    print(f"Rover lat/lon: ({rover_lat:.3f}, {rover_lon:.3f}) deg")
    print("R_true (Mars→Cam):\n", R_true)
    print("t_true (Mars frame, km):", t_true)

    # --- 4. Choose which landmarks are visible ---
    true_indices = rng.choice(range(n), size=min(num_true, n), replace=False)
    true_points_world = np.stack(
        [[c['x'], c['y'], c['elev']] for c in catalog if c['id'] in (true_indices + 1)],
        axis=0
    )

    # --- 5. Transform landmarks into camera frame ---
    points_cam = (R_true @ (true_points_world - t_true).T).T

    # --- 6. Keep points in front of camera and within field of view ---
    dir_cam = points_cam / np.linalg.norm(points_cam, axis=1, keepdims=True)
    cos_fov = np.cos(np.deg2rad(fov_deg / 2))
    in_fov = dir_cam[:, 2] > cos_fov
    points_cam = points_cam[in_fov]
    true_indices = true_indices[in_fov]

    # --- 7. Add angular noise ---
    sigma_rad = np.deg2rad(noise_deg)
    def perturb(v):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.normal(0, sigma_rad)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        Rn = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
        return (Rn @ v)
    observed_true = np.stack([perturb(v) for v in dir_cam], axis=0)

    # --- 8. Add false detections randomly distributed in FOV ---
    num_false = int(num_false)
    az = rng.uniform(-np.deg2rad(fov_deg/2), np.deg2rad(fov_deg/2), num_false)
    el = rng.uniform(-np.deg2rad(fov_deg/2), np.deg2rad(fov_deg/2), num_false)
    fx = np.cos(el) * np.cos(az)
    fy = np.cos(el) * np.sin(az)
    fz = np.sin(el)
    false_vecs = np.stack([fx, fy, fz], axis=1)

    # --- 9. Combine true + false observations ---
    observed_all = np.vstack([observed_true, false_vecs])

    return {
        "observed_vectors": observed_all,
        "true_indices": list(true_indices),
        "R_true": R_true,
        "t_true": t_true,
        "rover_latlon": (rover_lat, rover_lon),
        "observed_points_cam": points_cam
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