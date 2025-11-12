import numpy as np
import itertools
from .pyramids import build_kvector, query_kvector
from .geometry import kabsch_rotation, kabsch_pose, apply_rotation

def normalize_rows(V):
    return V / np.linalg.norm(V, axis=1, keepdims=True)

def edge(a, b):
    a = int(a); b = int(b)
    return frozenset((min(a, b), max(a, b)))

def to_edge_set(pairs):
    S = set()
    for a, b in pairs:
        S.add(edge(a, b))
    return S

def identify(catalog, observed_vectors, catalog_index=None,
             signature_tol_deg=1.0, ang_accept_deg=0.5, max_hypotheses=10000, R_true=None):
    """Identify observed stars via k-vector pyramid (triangle seed -> extend to 4th)."""
    if catalog_index is None or 'angles' not in catalog_index:
        catalog_index = build_kvector(catalog)

    # Normalize vectors
    cat_vecs_all = normalize_rows(np.stack([[c["vec"][0], c["vec"][1], c["vec"][2]] for c in catalog], axis=0))
    observed_vectors = normalize_rows(observed_vectors)

    # Precompute observed pair angles (radians)
    obs_pairs = []
    for i, j in itertools.combinations(range(len(observed_vectors)), 2):
        vi, vj = observed_vectors[i], observed_vectors[j]
        ang = np.arccos(np.clip(np.dot(vi, vj), -1.0, 1.0))
        obs_pairs.append((i, j, ang))

    obs_angle_map = {}
    for (i, j, ang) in obs_pairs:
        obs_angle_map[(i, j)] = ang
        obs_angle_map[(j, i)] = ang

    tol_rad = np.radians(signature_tol_deg)

    solutions = []
    best = None
    hypotheses_tested = 0

    # Iterate observed quadruples, but seed via the base triangle (oi,oj,ok)
    for (oi, oj, ok, ol) in itertools.combinations(range(len(observed_vectors)), 4):
        # Base triangle angles
        ang_ij = obs_angle_map[(oi, oj)]
        ang_ik = obs_angle_map[(oi, ok)]
        ang_jk = obs_angle_map[(oj, ok)]
        # Extension edges to 4th
        ang_il = obs_angle_map[(oi, ol)]
        ang_jl = obs_angle_map[(oj, ol)]
        ang_kl = obs_angle_map[(ok, ol)]

        # Query k-vector for the base triangle
        set_ij = to_edge_set(query_kvector(catalog_index, ang_ij, tol_rad))
        set_ik = to_edge_set(query_kvector(catalog_index, ang_ik, tol_rad))
        set_jk = to_edge_set(query_kvector(catalog_index, ang_jk, tol_rad))
        if not set_ij or not set_ik or not set_jk:
            continue

        # Candidate nodes from base triangle edges (small pool)
        tri_nodes = {n for e in set_ij | set_ik | set_jk for n in e}

        # Enumerate catalog triangles consistent with the base triangle
        for ci, cj, ck in itertools.combinations(tri_nodes, 3):
            if edge(ci, cj) not in set_ij or edge(ci, ck) not in set_ik or edge(cj, ck) not in set_jk:
                continue

            # Query k-vector for the extension edges
            set_il = to_edge_set(query_kvector(catalog_index, ang_il, tol_rad))
            set_jl = to_edge_set(query_kvector(catalog_index, ang_jl, tol_rad))
            set_kl = to_edge_set(query_kvector(catalog_index, ang_kl, tol_rad))
            if not set_il or not set_jl or not set_kl:
                continue

            # Candidate l nodes: nodes present in all three extension sets
            l_nodes = {n for e in set_il for n in e}
            l_nodes &= {n for e in set_jl for n in e}
            l_nodes &= {n for e in set_kl for n in e}
            if not l_nodes:
                continue

            # Try each candidate cl that closes the pyramid
            for cl in l_nodes:
                # Require 6/6 edge consistency (strict for clean/simulated data)
                if (edge(ci, cl) not in set_il or
                    edge(cj, cl) not in set_jl or
                    edge(ck, cl) not in set_kl):
                    continue

                # Permutation loop: match catalog quad to observed quad
                cat_nodes = (ci, cj, ck, cl)
                obs_nodes = (oi, oj, ok, ol)

                best_perm = None
                best_resid = np.inf
                best_R = None

                for perm in itertools.permutations(range(4)):
                    cat_quad = np.stack([cat_vecs_all[cat_nodes[p]] for p in perm], axis=0)
                    obs_quad = np.stack([observed_vectors[idx] for idx in obs_nodes], axis=0)

                    R_est, t_est = kabsch_pose(cat_quad, obs_quad)
                    # Reject reflections
                    if abs(np.linalg.det(R_est) - 1.0) > 1e-3:
                        continue

                    # Residual on the 4 points
                    cat_rot = apply_rotation(R_est, cat_quad)
                    resid = np.degrees(np.arccos(np.clip(np.sum(cat_rot * obs_quad, axis=1), -1, 1)))
                    mean4 = float(np.mean(resid))

                    if mean4 < best_resid:
                        best_resid = mean4
                        best_perm = perm
                        best_R = R_est

                # Discard if the 4-point residual is large
                if best_perm is None or best_resid > max(0.3, ang_accept_deg):
                    continue

                R_est = best_R

                # Apply rotation to entire catalog and verify one-to-one matches
                transformed_catalog = apply_rotation(R_est, cat_vecs_all)

                # Greedy one-to-one by observed stars (each observed gets one catalog)
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


def umeyama_2d(p, q, with_scale=False):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    assert p.shape == q.shape and p.shape[1] == 2
    N = p.shape[0]
    mu_p = p.mean(axis=0); mu_q = q.mean(axis=0)
    P = p - mu_p; Q = q - mu_q
    Sigma = (Q.T @ P) / N
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
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
