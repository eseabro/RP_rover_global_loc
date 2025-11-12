import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
import itertools, math, random, collections

def build_geometric_hash(catalog, k_neighbors=12, q=0.02, r_min=1.0, r_max=500.0):
    ids = [c['id'] for c in catalog]
    pos = np.array([[c['pos_x'], c['pos_y'], c['pos_z']] for c in catalog])
    mean_pos = pos.mean(axis=0)
    rel = pos - mean_pos
    pos2d = rel[:, :2]
    tree = cKDTree(pos2d)
    N = len(catalog)
    hash_table = collections.defaultdict(list)
    for i in range(N):
        kk = min(k_neighbors, N)
        dists, inds = tree.query(pos2d[i], k=kk)
        if kk ==1:
            inds = [inds]
        for a_idx in range(1, len(inds)):
            for b_idx in range(a_idx+1, len(inds)):
                j = inds[a_idx]
                k = inds[b_idx]
                if j == i or k == i or j == k:
                    continue
                A = pos2d[i]; B = pos2d[j]; C = pos2d[k]
                AB = B - A
                r = np.linalg.norm(AB)
                if r < r_min or r > r_max:
                    continue
                ex = AB / r
                ey = np.array([-ex[1], ex[0]])
                u = np.dot(C - A, ex)
                v = np.dot(C - A, ey)
                u_n = u / r; v_n = v / r
                # 2D cross (scalar) for area (A->B x A->C)
                cross2 = AB[0]*(C[1]-A[1]) - AB[1]*(C[0]-A[0])
                area = 0.5 * abs(cross2)
                if area < 1e-9 * (r**2):
                    continue
                key = (int(np.floor(u_n / q)), int(np.floor(v_n / q)), int(np.floor(np.log10(r+1e-9) / q)))
                hash_table[key].append((ids[i], ids[j], ids[k]))
    index = {'hash_table': hash_table, 'pos2d': pos2d, 'ids': ids, 'params': {'q': q, 'r_min': r_min, 'r_max': r_max}, 'mean_pos': mean_pos}
    return index

def query_geometric_hash(index, triple_local_pts, q_extra=0):
    A, B, C = triple_local_pts
    AB = B - A
    r = np.linalg.norm(AB)
    if r == 0:
        return []
    ex = AB / r
    ey = np.array([-ex[1], ex[0]])
    u = np.dot(C - A, ex); v = np.dot(C - A, ey)
    u_n = u / r; v_n = v / r
    q = index['params']['q']
    key0 = (int(np.floor(u_n / q)), int(np.floor(v_n / q)), int(np.floor(np.log10(r+1e-9) / q)))
    candidates = []
    for dx in range(-q_extra, q_extra+1):
        for dy in range(-q_extra, q_extra+1):
            key = (key0[0]+dx, key0[1]+dy, key0[2])
            candidates.extend(index['hash_table'].get(key, []))
    return candidates