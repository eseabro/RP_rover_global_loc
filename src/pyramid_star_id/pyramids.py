"""Pyramid signature generation and matching helpers."""
import itertools
import numpy as np
from .geometry import angular_distance_deg

def build_kvector(catalog):
    """
    Build k-vector index for pairwise angular separations in the catalog.
    Returns dict with:
      - 'pairs': list of (i,j) catalog indices
      - 'angles': sorted array of angular separations (radians)
      - 'kvec': k-vector array for O(1) range lookup
      - 'min_angle', 'max_angle': bounds
    """
    n = len(catalog)
    pairs = []
    angles = []
    vecs = np.stack([[c['x'], c['y'], c['elev']] for c in catalog], axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_unit = vecs / np.clip(norms, 1e-12, None)
    # compute all unique pairs
    for i, j in itertools.combinations(range(n), 2):
        v1, v2 = vecs_unit[i], vecs_unit[j]
        cosang = np.clip(np.dot(v1, v2), -1.0, 1.0)
        ang = np.arccos(cosang)
        pairs.append((i, j))
        angles.append(ang)

    angles = np.array(angles)
    pairs = np.array(pairs)

    # Sort by angle
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    pairs_sorted = pairs[sort_idx]

    # Build k-vector index (Mortari’s method)
    m = len(angles_sorted)
    kvec = np.zeros(m, dtype=int)
    if m > 1:
        for k in range(m):
            frac = (angles_sorted[k] - angles_sorted[0]) / (angles_sorted[-1] - angles_sorted[0] + 1e-12)
            kvec[k] = int(frac * (m - 1))

    return {
        "pairs": pairs_sorted,
        "angles": angles_sorted,
        "kvec": kvec,
        "min_angle": float(angles_sorted[0]),
        "max_angle": float(angles_sorted[-1])
    }

def query_kvector(kdata, angle, tol):
    """
    Query k-vector index for candidate pairs within [angle - tol, angle + tol].
    Returns list of (i,j) catalog indices.
    """
    angles = kdata['angles']
    pairs = kdata['pairs']

    lo = angle - tol
    hi = angle + tol

    # binary search for range
    lo_idx = np.searchsorted(angles, lo, side='left')
    hi_idx = np.searchsorted(angles, hi, side='right')

    return pairs[lo_idx:hi_idx]

def pyramid_signature_from_vectors(vecs):
    """vecs: (4,3) array. Returns sorted array of 6 angular distances (deg)."""
    angles = []
    for (i,j) in itertools.combinations(range(4), 2):
        angles.append(angular_distance_deg(vecs[i], vecs[j]))
    return np.sort(np.array(angles))

def precompute_catalog_pyramids(catalog):
    """Return list of dicts: {'indices': tuple, 'signature': np.array(6,)}"""
    n = len(catalog)
    indices = list(range(n))
    pyramids = []
    for comb in itertools.combinations(indices, 4):
        vecs = np.stack([catalog[i]['vec'] for i in comb], axis=0)
        sig = pyramid_signature_from_vectors(vecs)
        pyramids.append({"indices": comb, "signature": sig})
    return pyramids
