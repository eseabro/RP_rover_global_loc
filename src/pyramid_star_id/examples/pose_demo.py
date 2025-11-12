# Retry with fixes to latlon_to_unit_vector and small bugs.
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
import itertools, math, random, collections
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_matches(catalog_pts, observed_pts, inliers=None, matched_indices=None, title="Pyramid Matches"):
    """
    Visualize catalog points, observed points, and optionally matched inliers.

    Parameters
    ----------
    catalog_pts : (N,3) array
        Global catalog positions
    observed_pts : (M,3) array
        Rover observed positions
    inliers : list of int, optional
        Indices into observed_pts for points considered inliers
    matched_indices : list of int, optional
        Indices into catalog_pts corresponding to inliers
    title : str
        Plot title
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot catalog points
    ax.scatter(catalog_pts[:,0], catalog_pts[:,1], catalog_pts[:,2], c='blue', s=20, alpha=0.5, label='Catalog')
    
    # Plot all observed points
    ax.scatter(observed_pts[:,0], observed_pts[:,1], observed_pts[:,2], c='red', s=40, alpha=0.8, label='Observed')
    
    # If inliers/matches are provided, draw lines
    if inliers is not None and matched_indices is not None:
        for obs_idx, cat_idx in zip(inliers, matched_indices):
            p_obs = observed_pts[obs_idx]
            p_cat = catalog_pts[cat_idx]
            ax.plot([p_obs[0], p_cat[0]], [p_obs[1], p_cat[1]], [p_obs[2], p_cat[2]], c='green', linewidth=1)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=120)
    plt.show()

def normalize_rows(a):
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n

def umeyama_3d(p, q, with_scale=False):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    assert p.shape == q.shape and p.shape[1] == 3
    N = p.shape[0]
    mu_p = p.mean(axis=0)
    mu_q = q.mean(axis=0)
    P = p - mu_p
    Q = q - mu_q
    Sigma = (Q.T @ P) / N
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1,-1] = -1
    R = U @ S @ Vt
    if with_scale:
        var_p = np.sum(P**2) / N
        s = np.trace(np.diag(D) @ S) / var_p
    else:
        s = 1.0
    t = mu_q - s * (R @ mu_p)
    return s, R, t

def pyramid_descriptor(points, types=None):
    """Compute a translation + rotation + scale invariant pyramid descriptor."""
    assert points.shape[0] == 4
    dists = []
    for (i,j) in itertools.combinations(range(4), 2):
        dists.append(np.linalg.norm(points[i] - points[j]))
    dists = np.array(dists)
    base = np.min(dists) + 1e-9  # normalize by shortest edge
    ratios = np.sort(dists / base)
    desc = ratios
    if types is not None:
        # add normalized mean type values for discrimination
        desc = np.concatenate([desc, np.array([np.mean(types)])])
    return desc

def build_pyramid_catalog(catalog_points, catalog_types=None, report_every=100000):
    descs, pyramids = [], []
    count = 0
    total = math.comb(len(catalog_points), 4)
    
    for (i,j,k,l) in itertools.combinations(range(len(catalog_points)), 4):
        pts = np.stack([catalog_points[i], catalog_points[j], catalog_points[k], catalog_points[l]], axis=0)
        desc = pyramid_descriptor(pts, None if catalog_types is None else [catalog_types[i], catalog_types[j], catalog_types[k], catalog_types[l]])
        descs.append(desc)
        pyramids.append((i,j,k,l))
        
        count += 1
        if count % report_every == 0:
            print(f"Processed {count} / {total} pyramids ({count/total*100:.2f}%)", end='\r')
    
    descs = np.array(descs)
    tree = cKDTree(descs)
    return dict(points=catalog_points, pyramids=np.array(pyramids), descs=descs, tree=tree)

def query_pyramids(local_points, catalog, tol=0.02):
    matches = []
    for (i,j,k,l) in itertools.combinations(range(len(local_points)), 4):
        desc = pyramid_descriptor(np.stack([local_points[i], local_points[j], local_points[k], local_points[l]], axis=0))
        idxs = catalog['tree'].query_ball_point(desc, tol)
        for ci in idxs:
            matches.append(((i,j,k,l), catalog['pyramids'][ci]))
    return matches

def ransac_pose_pyramids(local_pts, catalog_pts, matches, n_iter=1000, inlier_thresh=10.0):
    rng = np.random.default_rng(3)
    best_inliers, best_R, best_t = [], None, None
    for _ in range(n_iter):
        if len(matches) < 4: break
        pyr_local, pyr_cat = matches[rng.integers(len(matches))]
        A = np.array([local_pts[i] for i in pyr_local])
        B = np.array([catalog_pts[j] for j in pyr_cat])
        s, R, t = umeyama_3d(A, B, with_scale=False)
        pred = (s * (local_pts @ R.T)) + t
        dists = np.linalg.norm(pred[:,None,:] - catalog_pts[None,:,:], axis=2)
        inliers = np.argwhere(np.min(dists,axis=1) < inlier_thresh).ravel()
        if len(inliers) > len(best_inliers):
            best_inliers, best_R, best_t, best_s = inliers, R, t, s
    return best_R, best_t, best_inliers

def latlon_to_unit_vector(lat_deg, lon_deg, radius=3389500.0):
    # Accept scalar lat/lon or arrays. Return unit vector (3,) and position (3,) in meters.
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    # if scalars, make them 0-d arrays but handle the broadcasting
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    vec = np.array([x, y, z], dtype=float)
    # if inputs were scalars, vec's shape is (3,), else might be shape (3, N) depending on broadcasting; ensure shape (3,) by flattening
    if vec.ndim > 1:
        vec = vec.reshape(3, -1).T  # (N,3)
    else:
        vec = vec.astype(float)
    pos = vec * radius
    if isinstance(lat_deg, (list, tuple, np.ndarray)):
        return vec, pos
    else:
        return vec.reshape(3,), pos.reshape(3,)

def generate_catalog(n=200, center_lat=0.0, center_lon=0.0, radius_km=1.0, seed=0):
    rng = np.random.RandomState(seed)
    types = ['rock','crater','chasm','boulder']
    catalog = []
    for i in range(n):
        r = radius_km * 1000.0 * np.sqrt(rng.rand())
        theta = rng.rand() * 2*np.pi
        dlat = (r * np.cos(theta)) / 111200.0
        dlon = (r * np.sin(theta)) / (111200.0 * np.cos(np.radians(center_lat)+1e-9))
        lat = center_lat + dlat
        lon = center_lon + dlon
        vec, pos = latlon_to_unit_vector(lat, lon)
        # vec,pos are 1D arrays
        t = rng.choice(types, p=[0.5, 0.2, 0.1, 0.2])
        rad = float(abs(rng.normal(loc=1.0 if t=='rock' else 10.0 if t=='crater' else 5.0, scale=0.5)))
        entry = {
            'id': i,
            'lat_deg': float(lat),
            'lon_deg': float(lon),
            'vec_x': float(vec[0]),
            'vec_y': float(vec[1]),
            'vec_z': float(vec[2]),
            'pos_x': float(pos[0]),
            'pos_y': float(pos[1]),
            'pos_z': float(pos[2]),
            'type': t,
            'radius': float(abs(rad))
        }
        catalog.append(entry)
    return catalog

def sample_observed(catalog, n_obs=10, seed=1, add_translation=True, pos_noise_m=0.5, ang_noise_deg=0.2):
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(catalog), size=min(n_obs, len(catalog)), replace=False)
    axis = normalize_rows(rng.randn(1,3))[0]
    angle = (rng.rand() - 0.5) * 2 * np.pi
    K = np.array([[0, -axis[2], axis[1]],[axis[2], 0, -axis[0]],[-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(angle)*K + (1-math.cos(angle))*(K@K)
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
        Rn = np.eye(3) + math.sin(ang)*K2 + (1-math.cos(ang))*(K2@K2)
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

def refine_similarity_3d(local_pts, catalog_pts, inliers, with_scale=False):
    if inliers is None or len(inliers) == 0:
        return None

    p = np.array([local_pts[i] for i in inliers])
    q = np.array([catalog_pts[i] for i in inliers])

    s_ref, R_ref, t_ref = umeyama_3d(p, q, with_scale=with_scale)

    # compute residual cost
    p_tr = (s_ref * (R_ref @ p.T)).T + t_ref
    cost = np.sum((p_tr - q)**2)

    return {'s': s_ref, 'R': R_ref, 't': t_ref, 'cost': float(cost), 'status': 1}

def refine_similarity_3d_nls(local_pts, catalog_pts, inliers, with_scale=False):
    p = np.array([local_pts[i] for i in inliers])
    q = np.array([catalog_pts[i] for i in inliers])

    s0, R0, t0 = umeyama_3d(p, q, with_scale=with_scale)
    rvec = R.from_matrix(R0).as_rotvec()
    x0 = np.hstack([rvec, t0])
    if with_scale:
        x0 = np.hstack([np.log(s0), x0])

    def residuals(x):
        if with_scale:
            s = np.exp(x[0])
            rvec = x[1:4]
            t = x[4:7]
        else:
            s = 1.0
            rvec = x[0:3]
            t = x[3:6]
        R_mat = R.from_rotvec(rvec).as_matrix()
        p_tr = (s * (R_mat @ p.T)).T + t
        return (p_tr - q).ravel()

    res = least_squares(residuals, x0, method='lm')
    if with_scale:
        s_ref = np.exp(res.x[0])
        rvec = res.x[1:4]
        t_ref = res.x[4:7]
    else:
        s_ref = 1.0
        rvec = res.x[0:3]
        t_ref = res.x[3:6]
    R_ref = R.from_rotvec(rvec).as_matrix()
    p_tr = (s_ref * (R_ref @ p.T)).T + t_ref
    cost = np.sum((p_tr - q)**2)
    return {'s': s_ref, 'R': R_ref, 't': t_ref, 'cost': float(cost), 'status': res.status}

def build_kvector(catalog):
    cat_vecs = np.stack([[c['vec_x'], c['vec_y'], c['vec_z']] for c in catalog], axis=0)
    cat_vecs = normalize_rows(cat_vecs)
    N = cat_vecs.shape[0]
    angles = []
    for i,j in itertools.combinations(range(N), 2):
        ang = math.acos(np.clip(np.dot(cat_vecs[i], cat_vecs[j]), -1.0, 1.0))
        angles.append((i,j,ang))
    angles_sorted = sorted(angles, key=lambda x: x[2])
    angle_vals = np.array([a[2] for a in angles_sorted])
    index = {'angles_sorted': angles_sorted, 'angle_vals': angle_vals, 'cat_vecs': cat_vecs}
    return index

def query_kvector(catalog_index, angle_rad, tol_rad):
    vals = catalog_index['angle_vals']
    lo = angle_rad - tol_rad; hi = angle_rad + tol_rad
    import bisect
    left = bisect.bisect_left(vals, lo)
    right = bisect.bisect_right(vals, hi)
    return catalog_index['angles_sorted'][left:right]

def edge(a,b):
    return (a,b) if a<=b else (b,a)

def to_edge_set(angle_entries):
    return {edge(i,j) for (i,j,ang) in angle_entries}

def kabsch_pose(A, B):
    A = np.asarray(A); B = np.asarray(B)
    muA = A.mean(axis=0); muB = B.mean(axis=0)
    AA = A - muA; BB = B - muB
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = muB - R @ muA
    return R, t

def apply_rotation(R, vecs):
    return (R @ vecs.T).T

def identify(catalog, observed_vectors, catalog_index=None,
             signature_tol_deg=1.0, ang_accept_deg=0.5, max_hypotheses=10000, R_true=None):
    if catalog_index is None or 'angles_sorted' not in catalog_index:
        catalog_index = build_kvector(catalog)
    cat_vecs_all = normalize_rows(np.stack([[c['vec_x'], c['vec_y'], c['vec_z']] for c in catalog], axis=0))
    observed_vectors = normalize_rows(np.asarray(observed_vectors))
    obs_pairs = []
    for i, j in itertools.combinations(range(len(observed_vectors)), 2):
        vi, vj = observed_vectors[i], observed_vectors[j]
        ang = math.acos(np.clip(np.dot(vi, vj), -1.0, 1.0))
        obs_pairs.append((i, j, ang))
    obs_angle_map = {}
    for (i, j, ang) in obs_pairs:
        obs_angle_map[(i, j)] = ang
        obs_angle_map[(j, i)] = ang
    tol_rad = np.radians(signature_tol_deg)
    ang_accept_deg = float(ang_accept_deg)
    solutions = []
    best = None
    hypotheses_tested = 0
    M = len(observed_vectors)
    for (oi, oj, ok, ol) in itertools.combinations(range(M), 4):
        ang_ij = obs_angle_map[(oi, oj)]
        ang_ik = obs_angle_map[(oi, ok)]
        ang_jk = obs_angle_map[(oj, ok)]
        ang_il = obs_angle_map[(oi, ol)]
        ang_jl = obs_angle_map[(oj, ol)]
        ang_kl = obs_angle_map[(ok, ol)]
        set_ij = to_edge_set(query_kvector(catalog_index, ang_ij, tol_rad))
        set_ik = to_edge_set(query_kvector(catalog_index, ang_ik, tol_rad))
        set_jk = to_edge_set(query_kvector(catalog_index, ang_jk, tol_rad))
        if not set_ij or not set_ik or not set_jk:
            continue
        tri_nodes = {n for e in set_ij | set_ik | set_jk for n in e}
        for ci, cj, ck in itertools.combinations(sorted(tri_nodes), 3):
            if edge(ci, cj) not in set_ij or edge(ci, ck) not in set_ik or edge(cj, ck) not in set_jk:
                continue
            set_il = to_edge_set(query_kvector(catalog_index, ang_il, tol_rad))
            set_jl = to_edge_set(query_kvector(catalog_index, ang_jl, tol_rad))
            set_kl = to_edge_set(query_kvector(catalog_index, ang_kl, tol_rad))
            if not set_il or not set_jl or not set_kl:
                continue
            l_nodes = {n for e in set_il for n in e}
            l_nodes &= {n for e in set_jl for n in e}
            l_nodes &= {n for e in set_kl for n in e}
            if not l_nodes:
                continue
            for cl in l_nodes:
                if (edge(ci, cl) not in set_il or
                    edge(cj, cl) not in set_jl or
                    edge(ck, cl) not in set_kl):
                    continue
                cat_nodes = (ci, cj, ck, cl)
                obs_nodes = (oi, oj, ok, ol)
                best_perm = None
                best_resid = np.inf
                best_R = None
                for perm in itertools.permutations(range(4)):
                    cat_quad = np.stack([cat_vecs_all[cat_nodes[p]] for p in perm], axis=0)
                    obs_quad = np.stack([observed_vectors[idx] for idx in obs_nodes], axis=0)
                    R_est, t_est = kabsch_pose(cat_quad, obs_quad)
                    cat_rot = apply_rotation(R_est, cat_quad)
                    dots = np.sum(cat_rot * obs_quad, axis=1)
                    dots = np.clip(dots, -1.0, 1.0)
                    resid = np.degrees(np.arccos(dots))
                    mean4 = float(np.mean(resid))
                    if mean4 < best_resid:
                        best_resid = mean4
                        best_perm = perm
                        best_R = R_est
                if best_perm is None or best_resid > max(0.3, ang_accept_deg):
                    continue
                R_est = best_R
                transformed_catalog = apply_rotation(R_est, cat_vecs_all)
                used_cat = set()
                matches = []
                for j, obs_v in enumerate(observed_vectors):
                    dots = transformed_catalog @ obs_v
                    angs = np.degrees(np.arccos(np.clip(dots, -1, 1)))
                    i = int(np.argmin(angs))
                    d = float(angs[i])
                    if d < ang_accept_deg and i not in used_cat:
                        matches.append((i, j, d))
                        used_cat.add(i)
                inlier_count = len(matches)
                mean_resid = float(np.mean([m[2] for m in matches])) if matches else None
                sol = {
                    "observed_quadruple": (oi, oj, ok, ol),
                    "catalog_quadruple": (ci, cj, ck, cl),
                    "R_est": R_est,
                    "t_est": t_est,
                    "inlier_count": inlier_count,
                    "mean_residual_deg": mean_resid,
                    "matches": matches
                }
                solutions.append(sol)
                if (best is None or
                    (inlier_count > best['inlier_count']) or
                    (inlier_count == best['inlier_count'] and (mean_resid is not None and
                        (best['mean_residual_deg'] is None or mean_resid < best['mean_residual_deg'])))):
                    best = sol
                hypotheses_tested += 1
                if hypotheses_tested > max_hypotheses:
                    break
            if hypotheses_tested > max_hypotheses:
                break
        if hypotheses_tested > max_hypotheses:
            break
    return {
        "candidates": hypotheses_tested,
        "solutions": solutions,
        "best_solution": best
    }

def _minimal_demo_pyramids():
    print("Generating catalog...")
    catalog = generate_catalog(n=30, center_lat=2.0, center_lon=137.0, radius_km=2.0, seed=42)
    print("Sampling observed features...")
    observed, gt = sample_observed(catalog, n_obs=20, seed=7, pos_noise_m=0.3, ang_noise_deg=0.15)
    
    cat_pos3d = np.array([[c['pos_x'], c['pos_y'], c['pos_z']] for c in catalog])
    local_pos3d = np.array([o['pos'] for o in observed])
    visualize_matches(cat_pos3d, local_pos3d, title="Before Alignment")


    # --- build the pyramid descriptor index ---
    print("Building pyramid catalog index...")
    catalog_pyramids = build_pyramid_catalog(cat_pos3d)

    # --- find candidate pyramid correspondences ---
    print("Matching observed pyramids...")
    matches = query_pyramids(local_pos3d, catalog_pyramids, tol=0.05)
    print(f"Found {len(matches)} pyramid correspondences")

    for i, (obs_quad, cand_indices) in enumerate(matches):
        print(f"{i}: obs_quad={obs_quad}, candidates={cand_indices}")


    # --- estimate pose via RANSAC ---
    print("Estimating pose via RANSAC...")
    R_est, t_est, inliers = ransac_pose_pyramids(local_pos3d, cat_pos3d, matches, n_iter=1000, inlier_thresh=150.0)
    print(f"RANSAC best: {len(inliers)} inliers, translation={t_est}, rotation=\n{R_est}")
    # --- refinement step (optional) ---
    if R_est is not None:
        refined = refine_similarity_3d_nls(local_pos3d, cat_pos3d, inliers, with_scale=False)
        print("Refined:", refined)
    else:
        refined = None
    # matched_catalog_indices = cat_pos3d[gt['selected_ids']]
    visualize_matches(cat_pos3d, local_pos3d, inliers, np.array(gt['selected_ids'], dtype=int), title="RANSAC Inliers")

    # --- diagnostics: differences ---
    if gt is not None and R_est is not None and t_est is not None:
        # translation difference vector
        t_diff = t_est - gt['t']
        t_err_norm = np.linalg.norm(t_diff)
        
        # rotation difference as angle (axis-angle)
        R_diff = R_est @ gt['R'].T
        angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        print(f"Translation error (m): {t_diff}, norm = {t_err_norm:.3f}")
        print(f"Rotation error (deg): {angle_deg:.3f}")
    return {
        'catalog': catalog,
        'observed': observed,
        'gt': gt,
        'R_est': R_est,
        't_est': t_est,
        'inliers': inliers,
        'refined': refined,
    }

_demo_results = _minimal_demo_pyramids()
_demo_results

