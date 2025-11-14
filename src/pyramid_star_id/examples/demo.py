"""Demo script showing how to use the package."""
from pyramid_star_id import build_geometric_hash_from_pts, build_catalog, identify_geometric, build_geometric_hash, simulate_observations_with_pose, save_catalog_csv, catalog_to_pts2d
import os
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_rover_scene(catalog, sim_result):
    """
    Plot the rover, catalog landmarks, and selected true + false landmarks.

    Args:
        catalog: list of dicts with 'x', 'y' entries.
        sim_result: dict returned from simulate_observations_with_pose().
    """
    # --- Extract data ---
    cat_xy = np.stack([[d['lon_deg'], d['lat_deg']] for d in catalog], axis=0)
    rover_pos = sim_result['t_rover']
    true_idx = sim_result['true_indices']
    num_false = sim_result["n_false"]

    # --- False points (if any) ---
    observed = sim_result.get('observed_vectors', None)

    # --- True landmarks (selected closest) ---
    n_true = len(true_idx)
    # true_points = observed[:n_true]
    # true_points = cat_xy[true_idx]
    true_points = observed[:n_true] @ sim_result['R_true'] + sim_result['t_rover']
    false_points = observed[n_true:]  if num_false > 0 else None

    # --- Plot ---
    plt.figure(figsize=(8, 8))
    plt.scatter(cat_xy[:, 0], cat_xy[:, 1], color='lightgray', label='Catalog landmarks')
    plt.scatter(true_points[:, 0], true_points[:, 1], color='dodgerblue', label='Selected landmarks', s=40)
    plt.scatter(rover_pos[0], rover_pos[1], color='red', marker='^', s=120, label='Rover position')
    if false_points is not None:
        plt.scatter(false_points[:, 0], false_points[:, 1], color='orange', marker='x', s=80, label='False landmarks')

    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title("Rover Scene: Catalog vs Selected Landmarks")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main():
    print('Building catalog...')
    catalog = build_catalog(n=100, seed=1)
    catalog_pts_2d = catalog_to_pts2d(catalog)
    os.makedirs('output', exist_ok=True)
    
    print('Saving catalog...')
    save_catalog_csv(catalog, 'output/catalog.csv')
    
    print('Precomputing catalog pyramids')
    ghash_index = build_geometric_hash_from_pts(catalog_pts_2d)
    
    print('Simulating observations...')
    sim = simulate_observations_with_pose(catalog, num_true=30, num_false=10, noise_deg=0.01, seed=2)
    obs = sim['observed_vectors'].astype(np.float32)
    true_indices = sim['true_indices']
    # plot_rover_scene(catalog=catalog, sim_result=sim)
    
    print('Identifying observed features...')
    result = identify_geometric(catalog_pts_2d, obs, hash_index=ghash_index)
    true_map = {obs_i: cat_i for obs_i, cat_i in enumerate(true_indices)}
    best = result['best_solution']

    if best:
        print('Matches (catalog_index, observed_index, residual):')
        for (cat_i, obs_i, resid) in best['matches']:
            truth = true_map.get(obs_i)
            correct = (truth == cat_i)
            mark = "✅" if correct else "❌"
            print(f'  cat={cat_i:3d}, obs={obs_i:2d}, resid={resid:6.3f}  {mark}')

        # For 2D: rotation angle and translation
        theta_deg = np.degrees(np.arctan2(best['R'][1,0], best['R'][0,0]))
        lon_est, lat_est = best["t"]  # translation interpreted as (lon, lat)
        print(f'Estimated rover pose: rotation={theta_deg:.2f}°, '
            f'longitude={lon_est:.3f}°, latitude={lat_est:.3f}°, '
            f'scale={best["s"]:.3f}')
    else:
        print('No solution found.')


if __name__ == '__main__':
    main()
