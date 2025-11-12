"""Demo script showing how to use the package."""
from pyramid_star_id import build_mars_catalog, simulate_observations, simulate_observations_with_pose, simulate_identity_observations, precompute_catalog_pyramids, identify, save_catalog_csv, build_kvector
import os

def main():
    print('Building catalog...')
    catalog = build_mars_catalog(n=60, seed=1)
    os.makedirs('output', exist_ok=True)
    print('Saving catalog...')
    save_catalog_csv(catalog, 'output/catalog.csv')
    print('Precomputing catalog pyramids')
    # catalog_pyrs = precompute_catalog_pyramids(catalog)
    kvec_index = build_kvector(catalog)
    print('Simulating observations...')
    sim = simulate_observations_with_pose(catalog, num_true=12, num_false=0, noise_deg=0.0, seed=2)
    # sim = simulate_identity_observations(catalog, num_true=10, seed=2)
    print("True indices:", sim['true_indices'])
    obs = sim['observed_vectors']
    true_indices = sim['true_indices']
    print('Identifying observed features...')
    print(obs)
    result = identify(catalog, obs, catalog_index=kvec_index,
                      signature_tol_deg=2.0, ang_accept_deg=2.0, R_true=sim['R_true'])
    true_map = {obs_i: cat_i for obs_i, cat_i in enumerate(true_indices)}
    best = result['best_solution']

    if best:
        print('Matches (catalog_index, observed_index, residual_deg):')
        for (cat_i, obs_i, resid) in best['matches']:
            truth = true_map.get(obs_i)
            correct = (truth == cat_i)
            mark = "✅" if correct else "❌"
            print(f'  cat={cat_i:3d}, obs={obs_i:2d}, resid={resid:6.3f}  {mark}')
        rover_position_global = - best["R_est"].T @ best["t_est"]
        print(f'Estimated rover position in global frame: {rover_position_global}')
    else:
        print('No solution found.')

if __name__ == '__main__':
    main()
