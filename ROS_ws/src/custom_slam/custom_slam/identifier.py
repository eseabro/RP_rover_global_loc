import itertools
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist, squareform

def quant(x, binsize=0.01):
    return int(np.floor(x / binsize))

def build_geometric_hash_fast(catalog_pts_2d, catalog_sizes=None, binsize=0.01, max_candidates_per_inv=40, k_neighbors=30):
    """
    Optimized Geometric Hash Builder using K-Nearest Neighbors.
    Complexity: O(N * k^2) instead of O(N^3).
    """
    pts = catalog_pts_2d
    n = len(pts)

    print(f"Building Hash for {n} points using {k_neighbors}-NN...")

    # 1. Build KDTree for fast neighbor lookup
    tree = KDTree(pts)
    
    # Query k+1 neighbors (because the point itself is the 1st neighbor)
    _, indices = tree.query(pts, k=min(n, k_neighbors + 1))

    # 2. Precompute distances
    if n < 5000:
        D = squareform(pdist(pts))
        use_precomputed = True
    else:
        use_precomputed = False

    table = {}
    seen_triangles = set() 

    # 3. Iterate through every point 'i' and form triangles with its neighbors
    for i in range(n):
        neighbors = indices[i, 1:]
        
        for j, k in itertools.combinations(neighbors, 2):
            
            tri_idxs = tuple(sorted((i, j, k)))
            
            if tri_idxs in seen_triangles:
                continue
            seen_triangles.add(tri_idxs)
            
            if use_precomputed:
                d_pq = D[tri_idxs[0], tri_idxs[1]]
                d_qr = D[tri_idxs[1], tri_idxs[2]]
                d_rp = D[tri_idxs[2], tri_idxs[0]]
                lengths = np.array([d_pq, d_qr, d_rp])
            else:
                p1, p2, p3 = pts[tri_idxs[0]], pts[tri_idxs[1]], pts[tri_idxs[2]]
                lengths = np.array([
                    np.linalg.norm(p1-p2),
                    np.linalg.norm(p2-p3),
                    np.linalg.norm(p3-p1)
                ])

            if np.any(lengths < 1e-9):
                continue
            
            idx_p, idx_q, idx_r = tri_idxs 
            vp, vq, vr = pts[idx_p], pts[idx_q], pts[idx_r]
            
            # NOTE: Assumes quantized_invariant is defined elsewhere in your file!
            inv = quantized_invariant(vp, vq, vr, binsize=binsize)
            if inv is None:
                continue

            e1 = np.linalg.norm(vp - vq) # Opp r
            e2 = np.linalg.norm(vq - vr) # Opp p
            e3 = np.linalg.norm(vr - vp) # Opp q
            
            edges = [ (e1, idx_r), (e2, idx_p), (e3, idx_q) ]
            edges.sort(key=lambda x: x[0])
            
            # These are the final, sorted triangle indices
            ci, cj, ck = edges[0][1], edges[1][1], edges[2][1]

            bucket = table.setdefault(inv, [])
            if len(bucket) < max_candidates_per_inv:
                # --- MODIFICATION 2: Store the sizes alongside the indices! ---
                if catalog_sizes is not None:
                    # Calculate Area (Width * Length) for each vertex using the canonical order
                    area_i = catalog_sizes[ci][0] * catalog_sizes[ci][1]
                    area_j = catalog_sizes[cj][0] * catalog_sizes[cj][1]
                    area_k = catalog_sizes[ck][0] * catalog_sizes[ck][1]
                    
                    # Store as a 6-item tuple: (idx1, idx2, idx3, area1, area2, area3)
                    bucket.append((ci, cj, ck, area_i, area_j, area_k))
                else:
                    # Fallback if no sizes are provided (stores standard 3-item tuple)
                    bucket.append((ci, cj, ck))
    # After building hash
    for key in list(table.keys()):
        if len(table[key]) > 50:
            table[key] = table[key][:50]
            
    return table


def greedy_unique_matches(dists, idxs, eps):
    """
    Given dists: (N,) distances from transformed observations to nearest catalog points
          idxs:  (N,) nearest catalog indices for each obs
    Return:
        inlier_mask: boolean mask of chosen inliers (obs -> unique catalog)
        chosen_cat_for_obs: length-N array of catalog indices (or -1)
        chosen_distances: length-N array of distances (or np.inf)
    Greedy policy: sort candidate obs by distance ascending, assign each obs
    only if its catalog wasn't already taken and distance <= eps.
    """
    N = len(idxs)
    chosen_cat_for_obs = -np.ones(N, dtype=int)
    chosen_distances = np.full(N, np.inf)
    inlier_mask = np.zeros(N, dtype=bool)

    # Create candidate list of (dist, obs_idx, cat_idx)
    cand = [(float(dists[i]), i, int(idxs[i])) for i in range(N) if dists[i] <= eps]
    cand.sort(key=lambda x: x[0])

    taken_cats = set()
    for dist, obs_i, cat_i in cand:
        if dist <= eps and cat_i not in taken_cats:
            taken_cats.add(cat_i)
            chosen_cat_for_obs[obs_i] = cat_i
            chosen_distances[obs_i] = dist
            inlier_mask[obs_i] = True

    return inlier_mask, chosen_cat_for_obs, chosen_distances


def score_transform_with_rms(obs_pts, cat_pts, s, R, t, tree=None, eps=10.0, return_assigned=False, obs_sizes=None, cat_sizes=None, size_tol=0.5):
    """
    Nearest-neighbor scoring but enforce unique correspondences greedily.
    Returns: inlier_count, inlier_mask, rms, all_dists, assigned_idxs
    """
    X = apply_similarity(obs_pts, s, R, t)
    if tree is None:
        tree = KDTree(cat_pts)
    dists, idxs = tree.query(X, k=1)
    dists = dists[:, 0]
    idxs = idxs[:, 0]
    
    # --- THE FIX: Disqualify Size Mismatches! ---
    if obs_sizes is not None and cat_sizes is not None:
        for i in range(len(obs_pts)):
            c_idx = idxs[i]
            # Scale the local area by s^2 so it mathematically matches the global map
            o_area = (obs_sizes[i][0] * obs_sizes[i][1]) * (s**2)
            c_area = cat_sizes[c_idx][0] * cat_sizes[c_idx][1]
            
            linear_ratio = np.sqrt(o_area / (c_area + 1e-6))
            if not (1.0 - size_tol < linear_ratio < 1.0 + size_tol):
                dists[i] = np.inf  # Force RANSAC to throw this match in the trash!
    # --------------------------------------------

    inlier_mask, chosen_cat_for_obs, chosen_dists = greedy_unique_matches(dists, idxs, eps)

    inlier_count = int(inlier_mask.sum())
    if inlier_count > 0:
        rms = np.sqrt(np.mean(chosen_dists[inlier_mask] ** 2))
    else:
        rms = np.inf

    if return_assigned:
        return inlier_count, inlier_mask, rms, chosen_dists, chosen_cat_for_obs
    else:
        return inlier_count, inlier_mask, rms, dists, chosen_cat_for_obs

def invariant_neighbors(inv, radius=1):
    """
    Generate neighboring invariant keys within ±radius for each dimension.
    """
    if inv is None:
        return []
    dims = len(inv)
    deltas = list(itertools.product(range(-radius, radius + 1), repeat=dims))
    return [tuple(inv[i] + d[i] for i in range(dims)) for d in deltas]


def apply_similarity(pts, s, R, t):
    return (s * (pts @ R.T)) + t

def refine_similarity(obs_pts, cat_pts, inliers_mask, s, R, t, tree=None, eps_refine=10.0):
    """
    Refit similarity transform using unique correspondences from the current transform.
    Steps:
      - Transform all observations using current transform
      - Find greedy unique NN matches within eps_refine
      - Fit Procrustes on matched pairs (obs -> cat)
    """
    _, inliers, _, _, assigned = score_transform_with_rms(
         obs_pts, cat_pts, s, R, t, tree, eps_refine, return_assigned=True
    )

    idxs_obs = np.where(inliers)[0]
    if len(idxs_obs) < 2:
        return s, R, t

    A = obs_pts[idxs_obs]
    B = cat_pts[assigned[idxs_obs]]

    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    A0 = A - ca
    B0 = B - cb
    na = np.linalg.norm(A0)
    nb = np.linalg.norm(B0)
    if na < 1e-12 or nb < 1e-12:
        return s, R, t
    s_new = nb / na
    H = A0.T @ B0
    U, _, Vt = np.linalg.svd(H)
    R_new = Vt.T @ U.T
    # if np.linalg.det(R_new) < 0:
    #     Vt[-1, :] *= -1
    #     R_new = Vt.T @ U.T
    t_new = cb - s_new * (R_new @ ca)

    return s_new, R_new, t_new


def estimate_similarity_from_triangle(A, B):
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ca, B - cb

    na = np.linalg.norm(A0)
    nb = np.linalg.norm(B0)
    if na < 1e-9 or nb < 1e-9:
        return None
    s = nb / na

    # Correct Kabsch: H = A0^T B0, U S Vt = svd(H), R = Vt.T @ U.T
    H = A0.T @ B0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # # Ensure proper rotation (det = +1)
    # if np.linalg.det(R) < 0:
    #     # fix reflection: flip last column of Vt (equivalently multiply V by diag(1, ..., -1))
    #     Vt[-1, :] *= -1
    #     R = Vt.T @ U.T

    t = cb - s * (R @ ca)
    return s, R, t


# --- Replace quantized_invariant with the full 5-dim descriptor ---
def quantized_invariant(p, q, r, binsize=0.01):
    """
    Permutation- and scale-invariant triangle descriptor built from sorted side lengths:
    - ratio1 = a/c, ratio2 = b/c
    - angle_a, angle_b: angles opposite sides a and b (via law of cosines, normalized by pi)
    - area_norm = area / c^2 (Heron’s formula for area, then normalize)
    Returns a 5-tuple of quantized integers.
    """
    # Side lengths
    d1 = np.linalg.norm(p - q)
    d2 = np.linalg.norm(q - r)
    d3 = np.linalg.norm(r - p)

    # Reject degenerate triangles early
    lengths = np.array([d1, d2, d3], dtype=np.float64)
    if np.any(lengths < 1e-9):
        return None

    # Sort sides so a <= b <= c (perm-invariant base)
    a, b, c = np.sort(lengths)

    # Ratios (scale-invariant)
    ratio1 = a / c
    ratio2 = b / c

    # Internal angles opposite a and b using sorted sides
    def safe_arccos(num, den):
        if den < 1e-12:
            return 0.0
        x = np.clip(num / den, -1.0, 1.0)
        return np.arccos(x)


    angle_a = safe_arccos(b**2 + c**2 - a**2, 2.0 * b * c) / np.pi  # normalize to [0,1]
    angle_b = safe_arccos(a**2 + c**2 - b**2, 2.0 * a * c) / np.pi

    # Area via Heron’s formula (perm-invariant), then normalize by c^2
    s = 0.5 * (a + b + c)  # semiperimeter
    area_sq = max(s * (s - a) * (s - b) * (s - c), 0.0)
    area = np.sqrt(area_sq)
    area_norm = area / (c**2 + 1e-12)

    return (
        quant(ratio1, binsize),
        quant(ratio2, binsize),
        quant(angle_a, binsize),
        quant(angle_b, binsize),
        # quant(area_norm, binsize),
    )

# --- helper: canonical vertex ordering for a triangle ---
def canonical_triangle_vertex_order(pts3):
    """
    Given pts3: (3,2) array for vertices [p,q,r] in that index order,
    return an index ordering [idx_opposite_a, idx_opposite_b, idx_opposite_c]
    where a<=b<=c are sorted side lengths and idx_opposite_* are the indices
    of the vertices opposite those sides.
    This ordering is deterministic and maps to the invariant's sorted-side convention.
    """
    p, q, r = pts3[0], pts3[1], pts3[2]

    d_pq = np.linalg.norm(p - q)
    d_qr = np.linalg.norm(q - r)
    d_rp = np.linalg.norm(r - p)

    lengths_with_opposite = [
        (d_pq, 2),
        (d_qr, 0),
        (d_rp, 1)
    ]
    lengths_with_opposite.sort(key=lambda x: x[0])  # ascending a<=b<=c
    ordered_vertices = [t[1] for t in lengths_with_opposite]
    return ordered_vertices  # length 3 list of indices in {0,1,2}


def identify_geometric(sim_result, catalog_dict,
                       hash_index=None,
                       eps=1.0,
                       binsize=0.01,
                       ransac_iters=4000,
                       inv_neighbor_radius=1,
                       max_candidates_per_inv=40,
                       min_seed_inliers=5,
                       early_exit_fraction=1.0,
                       size_tolerance=0.5):
    
    # 1. Clean Extraction: Trust the dictionary structure
    observed_pts_2d = sim_result['observed_vectors']
    observed_sizes = sim_result.get('observed_sizes') # May be None
    
    catalog_pts_2d = catalog_dict['catalog_vectors']
    catalog_sizes = catalog_dict.get('catalog_sizes') # May be None

    # 2. Build hash once if missing
    if hash_index is None:
        hash_index = build_geometric_hash_fast(catalog_pts_2d, catalog_sizes, binsize)

    # 3. Setup RANSAC
    # obs_indices = np.arange(len(observed_pts_2d))
    # all_obs_tris = list(itertools.combinations(obs_indices, 3))
    # cat_tree = KDTree(catalog_pts_2d)
    # rng = np.random.RandomState() # Truly random now

    # best = None
    # eps_init = max(eps * 3.0, 5.0) # Lowered initial coarse eps

    # for i in range(ransac_iters):
        # Select random triangle
        # tri_idx = all_obs_tris[rng.randint(0, len(all_obs_tris))]
        # obs_pts3 = observed_pts_2d[list(tri_idx)]
    # 3. Setup RANSAC
    num_local_rocks = len(observed_pts_2d)
    cat_tree = KDTree(catalog_pts_2d)
    obs_tree = KDTree(observed_pts_2d)
    rng = np.random.RandomState() # <-- Added the 42 seed for deterministic debugging!

    best = None
    # eps_init = max(eps * 3.0, 5.0) # Lowered initial coarse eps
    eps_init = eps * 3

    #---> FAST FIX 1: Pre-calculate all local neighbors ONCE <---
    k_search = min(num_local_rocks, 15)
    _, all_neighbors = obs_tree.query(observed_pts_2d, k=k_search)

    for i in range(ransac_iters):
        # --- NEW RANSAC SAMPLING: O(1) Lookup ---
        anchor_idx = rng.randint(0, num_local_rocks)
        
        # Grab precomputed neighbors (skip index 0, which is the rock itself)
        neighbors = all_neighbors[anchor_idx, 1:] 
        
        if len(neighbors) < 2: continue
        
        # Faster than rng.choice(..., replace=False) for small arrays
        idx1, idx2 = rng.choice(len(neighbors), 2, replace=False)
        tri_idx = [anchor_idx, neighbors[idx1], neighbors[idx2]]
        
        obs_pts3 = observed_pts_2d[tri_idx]
        
        # --- THE FIX: O(1) Memoryless Random Sampling ---
        # Instantly pick 3 unique indices directly from the array length
        # tri_idx = rng.choice(num_local_rocks, 3, replace=False)
        # obs_pts3 = observed_pts_2d[tri_idx]
        
        d1 = np.linalg.norm(obs_pts3[0] - obs_pts3[1])
        d2 = np.linalg.norm(obs_pts3[1] - obs_pts3[2])
        d3 = np.linalg.norm(obs_pts3[2] - obs_pts3[0])
        
        # Skipping Bad Triangles
        lengths = np.array([d1, d2, d3])
        a, b, c = np.sort(lengths)

        # Reject degenerate shapes
        if a / c < 0.15: continue
        if c < 2.0: continue
        
        
        # Find canonical order for deterministic matching
        obs_order = canonical_triangle_vertex_order(obs_pts3)
        o_a, o_b, o_c = [tri_idx[idx] for idx in obs_order]
        A = observed_pts_2d[[o_a, o_b, o_c]]

        # Generate hash key
        inv = quantized_invariant(A[0], A[1], A[2], binsize=binsize)
        if inv is None: continue
        
        # Pre-calculate areas for size filtering
        o_areas = None
        if observed_sizes is not None:
            o_areas = [observed_sizes[o_a][0]*observed_sizes[o_a][1], 
                       observed_sizes[o_b][0]*observed_sizes[o_b][1], 
                       observed_sizes[o_c][0]*observed_sizes[o_c][1]]

        # Lookup Shape Candidates
        candidate_triangles = []
        for key in invariant_neighbors(inv, radius=inv_neighbor_radius):
            tris = hash_index.get(key)
            if not tris: continue
            
            for tri_data in tris:
                # Optimized Size Filtering: Checks before any matrix math
                if len(tri_data) == 6 and o_areas:
                    ci, cj, ck, *c_areas = tri_data
                    # Take the square root here too!
                    linear_ratios = [np.sqrt(o_areas[i] / (c_areas[i] + 1e-6)) for i in range(3)]
                    if any(not (1.0 - size_tolerance < r < 1.0 + size_tolerance) for r in linear_ratios):
                        continue
                    candidate_triangles.append((ci, cj, ck))
                else:
                    candidate_triangles.append(tri_data[:3])

        if not candidate_triangles: continue
        rng.shuffle(candidate_triangles)

        # Transformation loop
        for (ci, cj, ck) in candidate_triangles[:max_candidates_per_inv]:
            B = catalog_pts_2d[[ci, cj, ck]]
            est = estimate_similarity_from_triangle(A, B)
            if not est or not (0.8 <= est[0] <= 1.2): continue
            
            s, R, t = est
            

            # Scoring: Coarse
            in_cnt_init, inliers_init, _, _, _ = score_transform_with_rms(
                observed_pts_2d, catalog_pts_2d, s, R, t, cat_tree, eps=eps_init,
                obs_sizes=observed_sizes, cat_sizes=catalog_sizes, size_tol=size_tolerance)

            if in_cnt_init < min_seed_inliers: continue

            # Refinement
            s_ref, R_ref, t_ref = refine_similarity(observed_pts_2d, catalog_pts_2d, inliers_init, s, R, t, cat_tree)

            if not (0.9 < s_ref < 1.1):
                continue
            
            # Scoring: Final with unique assignment
            in_cnt, inliers, rms, dists, assigned_idxs = score_transform_with_rms(
                observed_pts_2d, catalog_pts_2d, s_ref, R_ref, t_ref, cat_tree, eps=eps, return_assigned=True,
                obs_sizes=observed_sizes, cat_sizes=catalog_sizes, size_tol=size_tolerance)

            if in_cnt < min_seed_inliers: continue
            if rms > 0.5: continue

            # CONDENSED: Build matches directly from the scoring indices
            # No need for the extra cat_tree.query here anymore!
            matches = [(assigned_idxs[i], i, dists[i]) for i in np.where(inliers)[0]]

            if best is None or (in_cnt > best['inlier_count'] or (in_cnt == best['inlier_count'] and rms < best['rms'])):
                best = {
                    's': s_ref, 'R': R_ref, 't': t_ref,
                    'inlier_count': in_cnt, 'rms': rms,
                    'inliers': inliers, 'matches': matches
                }

            # Early Exit check
            if best and best['inlier_count'] >= early_exit_fraction * (len(observed_pts_2d)):
                return {'best_solution': best, 'iters': i+1, 'early_exit': True}

    return {'best_solution': best, 'iters': ransac_iters, 'early_exit': False}