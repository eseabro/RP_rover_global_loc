import numpy as np
import itertools

def catalog_to_pts2d(catalog):
    return np.stack([[c['x'], c['y']] for c in catalog], axis=0).astype(np.float32)

def triangle_invariants(p1, p2, p3):
    d12 = np.linalg.norm(p1 - p2)
    d13 = np.linalg.norm(p1 - p3)
    d23 = np.linalg.norm(p2 - p3)
    if min(d12, d13, d23) < 1e-9:
        return None
    ratios = sorted([d12/d13, d23/d13])
    return (round(ratios[0], 3), round(ratios[1], 3))

def build_geometric_hash_from_pts(catalog_pts_2d):
    table = {}
    n = len(catalog_pts_2d)
    for i, j, k in itertools.combinations(range(n), 3):
        inv = triangle_invariants(catalog_pts_2d[i], catalog_pts_2d[j], catalog_pts_2d[k])
        if inv is None:
            continue
        table.setdefault(inv, []).append((i, j, k))
    return table

def catalog_to_pts2d(catalog):
    # If your catalog entries have x,y
    return np.stack([[c['x'], c['y']] for c in catalog], axis=0).astype(np.float32)

def apply_similarity(pts, s, R, t):
    return (s * (pts @ R.T)) + t

from sklearn.neighbors import KDTree

def score_transform(obs_pts, cat_pts, s, R, t, eps=2.0):
    X = apply_similarity(obs_pts, s, R, t)
    tree = KDTree(cat_pts)
    dists, _ = tree.query(X, k=1)
    inliers = (dists[:,0] <= eps)
    return int(inliers.sum()), inliers

def refine_similarity(obs_pts, cat_pts, inliers, s, R, t):
    # Build matched pairs using nearest neighbors for inliers
    X = apply_similarity(obs_pts[inliers], s, R, t)
    tree = KDTree(cat_pts)
    _, idxs = tree.query(X, k=1)
    Y = cat_pts[idxs[:,0]]

    ca, cb = X.mean(axis=0), Y.mean(axis=0)
    X0, Y0 = X - ca, Y - cb
    na, nb = np.linalg.norm(X0), np.linalg.norm(Y0)
    if na < 1e-9 or nb < 1e-9:
        return s, R, t
    s_new = nb / na
    H = X0.T @ Y0
    U, _, Vt = np.linalg.svd(H)
    R_new = U @ Vt
    if np.linalg.det(R_new) < 0:
        Vt[-1, :] *= -1
        R_new = U @ Vt
    t_new = cb - s_new * (R_new @ ca)
    return s_new, R_new, t_new

def estimate_similarity_from_triangle(A, B):
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ca, B - cb

    # Step 2: scale from Frobenius norms
    na = np.linalg.norm(A0)
    nb = np.linalg.norm(B0)
    if na < 1e-9 or nb < 1e-9:
        return None
    s = nb / na

    # Step 3: rotation via 2D Procrustes
    H = A0.T @ B0
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # Step 4: translation
    t = cb - s * (R @ ca)

    return s, R, t

def identify_geometric(catalog_pts_2d, observed_pts_2d,
                       hash_index=None, eps=2.0, max_hypotheses=20000):
    if not isinstance(catalog_pts_2d, np.ndarray) or catalog_pts_2d.shape[1] != 2:
        raise ValueError("catalog_pts_2d must be a (N,2) ndarray")
    if not isinstance(observed_pts_2d, np.ndarray) or observed_pts_2d.shape[1] != 2:
        raise ValueError("observed_pts_2d must be a (M,2) ndarray")

    if hash_index is None:
        hash_index = build_geometric_hash_from_pts(catalog_pts_2d)

    best = None
    hypotheses = 0

    for oi, oj, ok in itertools.combinations(range(len(observed_pts_2d)), 3):
        inv = triangle_invariants(observed_pts_2d[oi], observed_pts_2d[oj], observed_pts_2d[ok])
        if inv is None or inv not in hash_index:
            continue

        for (ci, cj, ck) in hash_index[inv]:
            # Ensure indices are valid
            if max(ci, cj, ck) >= len(catalog_pts_2d):
                continue

            A = np.stack([observed_pts_2d[oi], observed_pts_2d[oj], observed_pts_2d[ok]], axis=0)
            B = np.stack([catalog_pts_2d[ci], catalog_pts_2d[cj], catalog_pts_2d[ck]], axis=0)

            est = estimate_similarity_from_triangle(A, B)
            if est is None:
                continue
            s, R, t = est

            in_cnt, inliers = score_transform(observed_pts_2d, catalog_pts_2d, s, R, t, eps=eps)
            if best is None or in_cnt > best['inlier_count']:
                s_ref, R_ref, t_ref = refine_similarity(observed_pts_2d, catalog_pts_2d, inliers, s, R, t)
                in_cnt_ref, inliers_ref = score_transform(observed_pts_2d, catalog_pts_2d, s_ref, R_ref, t_ref, eps=eps)

                # Optional: collect matches (nearest neighbors for inliers)
                from sklearn.neighbors import KDTree
                X = apply_similarity(observed_pts_2d, s_ref, R_ref, t_ref)
                tree = KDTree(catalog_pts_2d)
                dists, idxs = tree.query(X, k=1)
                matches = [(int(idxs[i,0]), int(i), float(dists[i,0])) for i in range(len(X)) if inliers_ref[i]]

                best = {
                    's': s_ref, 'R': R_ref, 't': t_ref,
                    'inlier_count': in_cnt_ref,
                    'inliers': inliers_ref,
                    'matches': matches,
                    'seed_obs_triangle': (oi, oj, ok),
                    'seed_cat_triangle': (ci, cj, ck)
                }

            hypotheses += 1
            if hypotheses >= max_hypotheses:
                break
        if hypotheses >= max_hypotheses:
            break

    return {
        'candidates': hypotheses,
        'best_solution': best
    }

def identify():
    pass