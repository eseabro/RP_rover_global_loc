"""Vector math, rotations, and Kabsch (orthogonal Procrustes)"""
import numpy as np

def angular_distance_deg(v1, v2):
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def kabsch_rotation(A, B):
    """Estimate rotation R such that R @ A.T ~= B.T (A,B: Nx3)
    Returns 3x3 rotation matrix."""
    assert A.shape == B.shape
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    return R


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

def umeyama_transform(src, dst):
    """Estimate rotation and translation using Umeyama algorithm."""
    assert src.shape == dst.shape
    n = src.shape[0]
    mean_src = np.mean(src, axis=0)
    mean_dst = np.mean(dst, axis=0)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst
    cov = np.dot(dst_centered.T, src_centered) / n
    U, S, Vt = np.linalg.svd(cov)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)
    t = mean_dst - np.dot(R, mean_src)
    return R, t

def apply_transform(R, t, vectors):
    """Apply rotation and translation to a set of vectors."""
    return np.dot(vectors, R.T) + t

def apply_rotation(R, vecs):
    return (R @ vecs.T).T

def random_rotation_matrix(seed=None):
    rng = np.random.RandomState(seed)
    u1, u2, u3 = rng.rand(), rng.rand(), rng.rand()
    q = np.array([
        np.sqrt(1-u1) * np.sin(2*np.pi*u2),
        np.sqrt(1-u1) * np.cos(2*np.pi*u2),
        np.sqrt(u1) * np.sin(2*np.pi*u3),
        np.sqrt(u1) * np.cos(2*np.pi*u3)
    ])
    x,y,z,w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]
    ])
    return R
