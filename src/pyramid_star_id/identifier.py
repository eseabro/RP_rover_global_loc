import itertools
import time
import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm

def quant(x, binsize=0.01):
    return int(np.floor(x / binsize))


# Then in catalog_to_pts2d, convert to km at a reference latitude:
def catalog_to_pts2d(catalog, ref_lat=0.0):
    """Convert catalog to km using simple projection at reference latitude."""
    deg_to_km_lat = 111.0
    deg_to_km_lon = 111.0 * np.cos(np.deg2rad(ref_lat))
    
    pts = np.array([[c['lon_deg'], c['lat_deg']] for c in catalog], dtype=np.float32)
    pts_km = pts * np.array([deg_to_km_lon, deg_to_km_lat])
    
    return pts_km

from sklearn.neighbors import KDTree # Ensure this is imported

def build_geometric_hash_fast(catalog_pts_2d, binsize=0.01, max_candidates_per_inv=40, k_neighbors=15):
    """
    Optimized Geometric Hash Builder using K-Nearest Neighbors.
    Complexity: O(N * k^2) instead of O(N^3).
    
    Args:
        k_neighbors: Only form triangles with the k nearest rocks. 
                     This implicitly ignores triangles that are too large 
                     for the rover to see at once.
    """
    pts = catalog_pts_2d
    n = len(pts)

    print(f"Building Hash for {n} points using {k_neighbors}-NN...")

    # 1. Build KDTree for fast neighbor lookup
    tree = KDTree(pts)
    
    # Query k+1 neighbors (because the point itself is the 1st neighbor)
    # This returns indices of the closest k points for every point
    _, indices = tree.query(pts, k=min(n, k_neighbors + 1))

    # 2. Precompute distances (Only if N < 5000 to save RAM, otherwise calc on fly)
    if n < 5000:
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(pts))
        use_precomputed = True
    else:
        use_precomputed = False

    table = {}
    seen_triangles = set() # To avoid duplicates like (A,B,C) vs (B,A,C)

    # 3. Iterate through every point 'i' and form triangles with its neighbors
    for i in range(n):
        # Get neighbors of i (excluding i itself, which is at index 0)
        neighbors = indices[i, 1:]
        
        # Form triangles where 'i' is the anchor, plus 2 neighbors 'j' and 'k'
        for j, k in itertools.combinations(neighbors, 2):
            
            # Sort indices to ensure uniqueness
            # (We only want to hash triangle 1-5-9 once, not as 5-1-9, etc.)
            tri_idxs = tuple(sorted((i, j, k)))
            
            if tri_idxs in seen_triangles:
                continue
            seen_triangles.add(tri_idxs)
            
            # --- Invariant Logic (Same as before) ---
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
            
            # Find Canonical Order (a <= b <= c)
            # We must map the sorted edges back to the original vertex indices
            # Indices in tri_idxs are [A, B, C]
            # Edges are [AB, BC, CA]
            # Opposites: AB->C(2), BC->A(0), CA->B(1)
            
            # Re-calculate specific edges for canonical ordering logic
            # (The previous `lengths` array was just sorted arbitrarily or by index)
            # Let's be explicit to match `quantized_invariant` logic:
            
            # Vertices
            idx_p, idx_q, idx_r = tri_idxs # These are sorted by index, not geometry
            vp, vq, vr = pts[idx_p], pts[idx_q], pts[idx_r]
            
            # This function calculates the invariant AND handles the sorting internally
            # We just need to store the indices in the order that `quantized_invariant` implies?
            # Actually, your original logic stored (ci, cj, ck) based on side lengths.
            # We need to replicate that.
            
            inv = quantized_invariant(vp, vq, vr, binsize=binsize)
            if inv is None:
                continue

            # Determining Storage Order:
            # We need to store vertices [Opp_a, Opp_b, Opp_c]
            # Re-calc edges to find permutation
            e1 = np.linalg.norm(vp - vq) # Opp r
            e2 = np.linalg.norm(vq - vr) # Opp p
            e3 = np.linalg.norm(vr - vp) # Opp q
            
            # lengths_map = [(edge_len, opposite_vertex_idx)]
            # We want side lengths a <= b <= c
            edges = [ (e1, idx_r), (e2, idx_p), (e3, idx_q) ]
            edges.sort(key=lambda x: x[0])
            
            # Unpack sorted indices
            ci, cj, ck = edges[0][1], edges[1][1], edges[2][1]

            bucket = table.setdefault(inv, [])
            if len(bucket) < max_candidates_per_inv:
                bucket.append((ci, cj, ck))

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
    cand = [(float(dists[i]), i, int(idxs[i])) for i in range(N)]
    cand.sort(key=lambda x: x[0])

    taken_cats = set()
    for dist, obs_i, cat_i in cand:
        if dist <= eps and cat_i not in taken_cats:
            taken_cats.add(cat_i)
            chosen_cat_for_obs[obs_i] = cat_i
            chosen_distances[obs_i] = dist
            inlier_mask[obs_i] = True

    return inlier_mask, chosen_cat_for_obs, chosen_distances


def score_transform_with_rms(obs_pts, cat_pts, s, R, t, tree=None, eps=10.0, return_assigned=False):
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

    inlier_mask, chosen_cat_for_obs, chosen_dists = greedy_unique_matches(dists, idxs, eps)

    inlier_count = int(inlier_mask.sum())
    if inlier_count > 0:
        rms = np.sqrt(np.mean(chosen_dists[inlier_mask] ** 2))
    else:
        rms = np.inf

    # For ease of later processing: assigned_idxs gives -1 for unmatched obs
    # return inlier_count, inlier_mask, rms, dists, chosen_cat_for_obs
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
    if np.linalg.det(R_new) < 0:
        Vt[-1, :] *= -1
        R_new = Vt.T @ U.T
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

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        # fix reflection: flip last column of Vt (equivalently multiply V by diag(1, ..., -1))
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

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

    # angle opposite a
    angle_a = safe_arccos(b**2 + c**2 - a**2, 2.0 * b * c) / np.pi  # normalize to [0,1]
    # angle opposite b
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
        quant(area_norm, binsize),
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
    # pts3 assumed shape (3,2) corresponding to vertices [0,1,2] -> p,q,r
    p, q, r = pts3[0], pts3[1], pts3[2]
    # side lengths and which vertex they are opposite to:
    # d(p,q) is opposite r -> index 2
    d_pq = np.linalg.norm(p - q)
    # d(q,r) opposite p -> index 0
    d_qr = np.linalg.norm(q - r)
    # d(r,p) opposite q -> index 1
    d_rp = np.linalg.norm(r - p)

    lengths_with_opposite = [
        (d_pq, 2),
        (d_qr, 0),
        (d_rp, 1)
    ]
    lengths_with_opposite.sort(key=lambda x: x[0])  # ascending a<=b<=c
    # extract the vertex indices in order opposite a,b,c
    ordered_vertices = [t[1] for t in lengths_with_opposite]
    return ordered_vertices  # length 3 list of indices in {0,1,2}


# Ran in 90.5 seconds
from scipy.spatial.distance import pdist, squareform
def build_geometric_hash_from_pts(catalog_pts_2d, binsize=0.01, max_candidates_per_inv=40):
    """
    Very fast geometric hash builder (no Numba).
    Uses:
        - one pdist() for all pairwise distances
        - canonical triangle ordering
        - your full 5-d invariant
    """
    pts = catalog_pts_2d
    n = len(pts)

    # Precompute pairwise distances once
    D = squareform(pdist(pts))

    table = {}

    for i, j, k in itertools.combinations(range(n), 3):
        # Distances without recomputation
        d_pq = D[i, j]
        d_qr = D[j, k]
        d_rp = D[k, i]

        lengths = np.array([d_pq, d_qr, d_rp])
        if np.any(lengths < 1e-9):
            continue

        idxs = [i, j, k]
        opposite = [2, 0, 1]
        order = [idxs[opposite[idx]] for idx in np.argsort(lengths)]

        ci, cj, ck = order

        # Compute full 5-dim invariant (your existing function)
        inv = quantized_invariant(
            catalog_pts_2d[ci], catalog_pts_2d[cj], catalog_pts_2d[ck], binsize=binsize
        )
        if inv is None:
            continue

        # Store with cap
        bucket = table.setdefault(inv, [])
        if len(bucket) < max_candidates_per_inv:
            bucket.append((ci, cj, ck))

    return table

def identify_geometric(sim_result, catalog,
                       hash_index=None,
                       eps=8.0,
                       binsize=0.02,
                       ransac_iters=4000,
                       inv_neighbor_radius=1,
                       max_candidates_per_inv=40,
                       min_seed_inliers=3,
                       early_exit_fraction=0.70):
    """
    Robust geometric hashing + RANSAC landmark identification using full 5-dim invariant
    and canonical triangle vertex ordering so that triangle vertices correspond deterministically.
    """

    # --- Extract data ---
    if isinstance(catalog, list):
        catalog_pts_2d = catalog_to_pts2d(catalog)
    else:
        catalog_pts_2d = catalog

    observed_pts_2d = sim_result['observed_vectors']

    # Validate shapes (keep early errors)
    if not isinstance(catalog_pts_2d, np.ndarray) or catalog_pts_2d.shape[1] != 2:
        raise ValueError("catalog_pts_2d must be a (N,2) ndarray")
    if not isinstance(observed_pts_2d, np.ndarray) or observed_pts_2d.shape[1] != 2:
        raise ValueError("observed_pts_2d must be a (M,2) ndarray")

    # --- Build hash if missing ---
    if hash_index is None:
        start = time.perf_counter()
        hash_index = build_geometric_hash_fast(catalog_pts_2d, binsize=0.01, k_neighbors=20)
        end = time.perf_counter()
        print(f"Geometric hash built in {end - start:.3f} s")

    obs_indices = np.arange(len(observed_pts_2d))
    all_obs_tris = list(itertools.combinations(obs_indices, 3))
    # cap observed triangles for speed if huge
    if len(all_obs_tris) > 20000:
        rng_cap = np.random.RandomState(42)
        keep = rng_cap.choice(len(all_obs_tris), 20000, replace=False)
        all_obs_tris = [all_obs_tris[i] for i in keep]

    cat_tree = KDTree(catalog_pts_2d)
    rng = np.random.RandomState(123)

    best = None
    eps_init = max(eps * 3.0, 20.0)

    # RANSAC with tqdm
    for _ in tqdm(range(ransac_iters), desc="RANSAC"):

        # pick a random observed triangle (in original index order)
        oi, oj, ok = all_obs_tris[rng.randint(0, len(all_obs_tris))]
        obs_pts3 = np.stack([observed_pts_2d[oi], observed_pts_2d[oj], observed_pts_2d[ok]], axis=0)

        # canonicalize observed triangle vertex order
        obs_order = canonical_triangle_vertex_order(obs_pts3)
        o_a, o_b, o_c = [ (oi, oj, ok)[idx] for idx in obs_order ]
        A = np.stack([observed_pts_2d[o_a], observed_pts_2d[o_b], observed_pts_2d[o_c]], axis=0)

        # compute invariant on canonical ordering (permutation invariant anyway)
        inv = quantized_invariant(A[0], A[1], A[2], binsize=binsize)
        if inv is None:
            continue

        # invariant neighbor keys
        candidate_invs = invariant_neighbors(inv, radius=inv_neighbor_radius)

        # collect catalog triangle candidates (they were stored canonical-ordered)
        candidate_triangles = []
        for key in candidate_invs:
            tris = hash_index.get(key)
            if tris:
                # cap the number taken from this key
                candidate_triangles.extend(tris[:max_candidates_per_inv])

        if not candidate_triangles:
            continue

        rng.shuffle(candidate_triangles)

        # try each candidate catalog triangle
        for (ci, cj, ck) in candidate_triangles:
            # B is already canonical-ordered during hash build
            B = np.stack([catalog_pts_2d[ci], catalog_pts_2d[cj], catalog_pts_2d[ck]], axis=0)

            est = estimate_similarity_from_triangle(A, B)
            if est is None:
                continue
            s, Rm, t = est

            # sanity on scale
            if not (0.05 <= s <= 20.0):
                continue

            # coarse check with loose eps_init using RMS + unique matching
            in_cnt_init, inliers_init, rms_init, dists_init, idxs_init = \
                score_transform_with_rms(observed_pts_2d, catalog_pts_2d, s, Rm, t, cat_tree, eps=eps_init)

            if in_cnt_init < min_seed_inliers:
                continue

            # ensure seed vertices are among coarse inliers (use their canonical indices)
            seed_obs_indices = [o_a, o_b, o_c]
            if not (inliers_init[seed_obs_indices[0]] and inliers_init[seed_obs_indices[1]] and inliers_init[seed_obs_indices[2]]):
                continue

            # refine using unique NN assignment
            s_ref, R_ref, t_ref = refine_similarity(observed_pts_2d, catalog_pts_2d, inliers_init, s, Rm, t)

            # final scoring with true eps
            in_cnt, inliers, rms, dists, ransac_idxs = score_transform_with_rms(
                observed_pts_2d, catalog_pts_2d, s_ref, R_ref, t_ref, cat_tree, eps=eps)

            if in_cnt < min_seed_inliers:
                continue

            # ensure seed still inliers
            if not (inliers[seed_obs_indices[0]] and inliers[seed_obs_indices[1]] and inliers[seed_obs_indices[2]]):
                continue

            # build matches with greedy unique assignment
            X = apply_similarity(observed_pts_2d, s_ref, R_ref, t_ref)
            d_all, i_all = cat_tree.query(X, k=1)
            d_all = d_all[:, 0]
            i_all = i_all[:, 0]
            
            matches = [(ransac_idxs[i], i, dists[i]) for i in np.where(inliers)[0]]

            # accept improvement
            if best is None or (in_cnt > best['inlier_count'] or (in_cnt == best['inlier_count'] and rms < best.get('rms', np.inf))):
                best = {
                    's': s_ref, 'R': R_ref, 't': t_ref,
                    'inlier_count': in_cnt,
                    'rms': rms,
                    'inliers': inliers,
                    'matches': matches,
                    'seed_obs_triangle': (o_a, o_b, o_c),
                    'seed_cat_triangle': (ci, cj, ck),
                }

            # early exit if we've matched a large fraction of expected true obs
            if best is not None:
                n_obs = len(observed_pts_2d)
                expected_true = n_obs - sim_result.get('n_false', 0)
                if best['inlier_count'] >= early_exit_fraction * expected_true:
                    return {'best_solution': best}

    return {'best_solution': best}

