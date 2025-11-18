from pyramid_star_id import build_catalog, identify_geometric, simulate_observations_with_pose, catalog_to_pts2d, save_catalog_csv
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_rover_scene(catalog, sim_result):
    """
    Plot the rover, catalog landmarks, and selected true + false landmarks.
    Creates two plots: one in degrees and one in km.

    Args:
        catalog: list of dicts with 'lon_deg', 'lat_deg' entries.
        sim_result: dict returned from simulate_observations_with_pose().
    """
    # --- Extract data ---
    cat_xy = np.stack([[d['lon_deg'], d['lat_deg']] for d in catalog], axis=0)
    rover_pos = sim_result['t_rover']  # [lon, lat] in degrees
    true_idx = sim_result['true_indices']
    num_false = sim_result["n_false"]

    # --- False points (if any) ---
    observed = sim_result.get('lat_lon_vectors', None) @ sim_result['R_true'] + sim_result['t_rover']

    # --- True landmarks (selected closest) ---
    n_true = len(true_idx)
    true_points = observed[:n_true] 
    false_points = observed[n_true:] if num_false > 0 else None

    # --- PLOT 1: Degrees ---
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.scatter(cat_xy[:, 0], cat_xy[:, 1], color='lightgray', label='Catalog landmarks', s=80)
    ax1.scatter(true_points[:, 0], true_points[:, 1], color='dodgerblue', label='Selected landmarks', s=10)
    ax1.scatter(rover_pos[0], rover_pos[1], color='red', marker='^', s=120, label='Rover position')
    if false_points is not None:
        ax1.scatter(false_points[:, 0], false_points[:, 1], color='orange', marker='x', s=80, label='False landmarks')

    ax1.set_xlabel("Longitude (deg)")
    ax1.set_ylabel("Latitude (deg)")
    ax1.set_title("Rover Scene: Degrees")
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    # --- PLOT 2: Kilometers ---
    cat_xy_km = catalog_to_pts2d(catalog=catalog)

    # Get observed vectors in km (from simulator - rover's local frame)
    observed_km_local = sim_result['observed_vectors']

    # Convert rover position to km using SAME simple method as catalog
    ref_lat = 0.0
    deg_to_km_lat = 111.0
    deg_to_km_lon = 111.0 * np.cos(np.deg2rad(ref_lat))
    rover_pos_km = rover_pos * np.array([deg_to_km_lon, deg_to_km_lat])

    # Transform observations to global frame
    R_true = sim_result['R_true']
    observed_km_global = (observed_km_local @ R_true) + rover_pos_km

    # Rest of plotting...
    true_points_km = observed_km_global[:n_true]
    false_points_km = observed_km_global[n_true:] if num_false > 0 else None
    
    ax2.scatter(cat_xy_km[:, 0], cat_xy_km[:, 1], color='lightgray', label='Catalog landmarks', s=80)
    ax2.scatter(true_points_km[:, 0], true_points_km[:, 1], color='dodgerblue', label='Selected landmarks', s=10)
    ax2.scatter(rover_pos_km[0], rover_pos_km[1], color='red', marker='^', s=120, label='Rover position')
    if false_points_km is not None:
        ax2.scatter(false_points_km[:, 0], false_points_km[:, 1], color='orange', marker='x', s=80, label='False landmarks')

    ax2.set_xlabel("East (km)")
    ax2.set_ylabel("North (km)")
    ax2.set_title("Rover Scene: Kilometers (Global Frame)")
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    print("\nSample true point (km, global frame):", true_points_km[5])
    print("Sample catalog point (km, catalog frame):", cat_xy_km[5])
    print("Rover position (km, catalog frame):", rover_pos_km)
    
    plt.tight_layout()
    plt.show()

def main():
    print('Building catalog...')
    n = 200
    region = (-5, 5, -5, 5)  # (lat_min, lat_max, lon_min, lon_max)
    file_path = f'output/catalog_{n}_{region[1]-region[0]}_{region[3]-region[2]}.csv'
    if os.path.exists(file_path):
        print("File exists!")
    else:
        print("File does not exist.")
    catalog = build_catalog(n=n, seed=1, region=region)
    # catalog_pts_2d = catalog_to_pts2d(catalog)
    os.makedirs('output', exist_ok=True)    
    
    print('Simulating observations...')
    sim = simulate_observations_with_pose(catalog, num_true=20, num_false=15, noise_deg=0.02, seed=2)
    print(f"True rotation: {sim['R_true']}")
    print(f"True translation: {sim['t_rover']}")
    true_indices = sim['true_indices']
    # plot_rover_scene(catalog=catalog, sim_result=sim)
    
    result = identify_geometric(sim, catalog, hash_index=None, eps=1.0, binsize=0.01)
    true_map = {obs_i: cat_i for obs_i, cat_i in enumerate(true_indices)}
    best = result['best_solution']

    if best:
        print(f"Solution with {best['inlier_count']} inliers found!")
        print('Matches (catalog_index, observed_index, residual):')
        for (cat_i, obs_i, resid) in best['matches']:
            truth = true_map.get(obs_i)
            correct = (truth == cat_i)
            mark = "✅" if correct else "❌"
            print(f'  cat={cat_i:3d}, obs={obs_i:2d}, resid={resid:6.3f}  {mark}')

        # For 2D: rotation angle and translation
        theta_deg = np.degrees(np.arctan2(best['R'][1,0], best['R'][0,0]))
        
        # Convert translation from km back to degrees
        ref_lat = 0.0
        deg_to_km_lat = 111.0
        deg_to_km_lon = 111.0 * np.cos(np.deg2rad(ref_lat))
        
        lon_est = best["t"][0] / deg_to_km_lon
        lat_est = best["t"][1] / deg_to_km_lat
        
        print(f'Estimated rover pose: rotation={theta_deg:.2f}°, '
            f'longitude={lon_est:.3f}°, latitude={lat_est:.3f}°, '
            f'scale={best["s"]:.3f}')
        
        # Compare with ground truth
        print(f'\nGround truth: rotation={np.degrees(np.arctan2(sim["R_true"][1,0], sim["R_true"][0,0])):.2f}°, '
            f'longitude={sim["t_rover"][0]:.3f}°, latitude={sim["t_rover"][1]:.3f}°')
    else:
        print('No solution found.')


if __name__ == '__main__':
    main()
