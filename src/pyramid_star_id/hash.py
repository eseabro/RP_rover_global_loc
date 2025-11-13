import itertools
import numpy as np
def build_geometric_hash(catalog):
    hash_table = {}
    for i, j, k in itertools.combinations(range(len(catalog)), 3):
        p1, p2, p3 = np.array([catalog[i]['x'], catalog[i]['y']]), \
                     np.array([catalog[j]['x'], catalog[j]['y']]), \
                     np.array([catalog[k]['x'], catalog[k]['y']])
        d12 = np.linalg.norm(p1 - p2)
        d13 = np.linalg.norm(p1 - p3)
        d23 = np.linalg.norm(p2 - p3)
        ratios = tuple(sorted([d12/d13, d23/d13]))
        key = (round(ratios[0], 3), round(ratios[1], 3))
        hash_table.setdefault(key, []).append((i, j, k))
    return hash_table

def query_geometric_hash(hash_table, obs_points):
    matches = []
    for oi, oj, ok in itertools.combinations(range(len(obs_points)), 3):
        p1, p2, p3 = obs_points[oi], obs_points[oj], obs_points[ok]
        d12 = np.linalg.norm(p1 - p2)
        d13 = np.linalg.norm(p1 - p3)
        d23 = np.linalg.norm(p2 - p3)
        ratios = tuple(sorted([d12/d13, d23/d13]))
        key = (round(ratios[0], 3), round(ratios[1], 3))
        if key in hash_table:
            for cat_trip in hash_table[key]:
                matches.append((cat_trip, (oi, oj, ok)))
    return matches
