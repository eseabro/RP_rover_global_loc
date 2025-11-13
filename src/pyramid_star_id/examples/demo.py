"""Demo script showing how to use the package."""
from pyramid_star_id import build_geometric_hash_from_pts, build_catalog, identify_geometric, build_geometric_hash, simulate_observations_with_pose, save_catalog_csv, catalog_to_pts2d
import os
import numpy as np

def main():
    print('Building catalog...')
    catalog = build_catalog(n=60, seed=1)
    catalog_pts_2d = catalog_to_pts2d(catalog)
    os.makedirs('output', exist_ok=True)
    print('Saving catalog...')
    save_catalog_csv(catalog, 'output/catalog.csv')
    print('Precomputing catalog pyramids')
    # catalog_pyrs = precompute_catalog_pyramids(catalog)
    # ghash_index = build_geometric_hash(catalog)
    ghash_index = build_geometric_hash_from_pts(catalog_pts_2d)
    print('Simulating observations...')
    sim = simulate_observations_with_pose(catalog, num_true=30, num_false=0, noise_deg=0.0, seed=2)
    obs = sim['observed_vectors'].astype(np.float32)
    true_indices = sim['true_indices']
    print('Identifying observed features...')
    # print(obs)

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
