import itertools
import numpy as np
from math import radians, degrees
from collections import defaultdict
from utils import *
np.random.seed(42)

# Full pyramid identification pipeline
# - Generates a synthetic star catalog (RA, Dec, unit vectors)
# - Simulates an observed star set by rotating a subset of catalog stars and adding noise + false detections
# - Precomputes pyramid (4-star) signatures for the catalog
# - Matches observed pyramids to catalog pyramids by angular signature
# - Verifies candidate matches using Kabsch (orthogonal Procrustes) to estimate rotation and count inliers
# - Outputs best match, estimated rotation, and matched star IDs
#


# -------------------------
# 1) Build catalog
# -------------------------
catalog_size = 30  # adjust: small enough to precompute pyramids
# For realism, produce a non-uniform distribution: sample declination with sin distribution
ras = np.random.uniform(0, 360, size=catalog_size)
# sample dec with sin-weighting to be uniform on sphere: inverse transform
u = np.random.uniform(-1, 1, size=catalog_size)
decs = np.degrees(np.arcsin(u))
catalog_vectors = ra_dec_to_vector(ras, decs)
# Assign simple IDs and magnitudes (random)
ids = np.arange(1, catalog_size+1)
mags = np.round(np.random.uniform(1.0, 6.5, size=catalog_size), 2)

# Create catalog as a structured array/dict
catalog = [{
    "id": int(ids[i]),
    "ra_deg": float(ras[i]),
    "dec_deg": float(decs[i]),
    "vec": catalog_vectors[i],
    "mag": float(mags[i])
} for i in range(catalog_size)]

# -------------------------
# 2) Precompute catalog pyramid signatures
# -------------------------
# Pyramid: any 4-star combination -> 6 inter-star angles.
# We'll store signature as sorted vector of 6 angular distances (in degrees).
# Also keep the indices mapping for retrieval.

def pyramid_signature_from_vectors(vecs):
    # vecs: (4,3)
    angles = []
    for (i,j) in itertools.combinations(range(4), 2):
        angles.append(angular_distance_deg(vecs[i], vecs[j]))
    angles = np.sort(np.array(angles))
    return angles

# Precompute for all 4-combinations. This can be large if catalog_size grows.
catalog_indices = list(range(catalog_size))
catalog_pyramids = []  # list of dicts: {indices: (i,j,k,l), signature: np.array([...])}
print("Precomputing catalog pyramids (this may take a moment)...")
for comb in itertools.combinations(catalog_indices, 4):
    vecs = np.stack([catalog[i]["vec"] for i in comb], axis=0)
    sig = pyramid_signature_from_vectors(vecs)
    catalog_pyramids.append({"indices": comb, "signature": sig})
catalog_pyramids = np.array(catalog_pyramids, dtype=object)
print(f"Precomputed {len(catalog_pyramids)} catalog pyramids.")

# -------------------------
# 3) Simulate observed stars
# -------------------------
# Select a subset of catalog stars to be visible in the tracker
num_true_visible = 8  # number of real catalog stars observed
true_indices = np.random.choice(catalog_indices, size=num_true_visible, replace=False)
true_vectors_inertial = np.stack([catalog[i]["vec"] for i in true_indices], axis=0)

# Create a random rotation (attitude) to map inertial -> body frame
def random_rotation_matrix():
    # Using uniform random quaternion
    u1, u2, u3 = np.random.rand(), np.random.rand(), np.random.rand()
    q = np.array([
        np.sqrt(1-u1) * np.sin(2*np.pi*u2),
        np.sqrt(1-u1) * np.cos(2*np.pi*u2),
        np.sqrt(u1) * np.sin(2*np.pi*u3),
        np.sqrt(u1) * np.cos(2*np.pi*u3)
    ])
    # quaternion to rotation matrix (w last): q = [x,y,z,w]
    x,y,z,w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]
    ])
    return R

R_true = random_rotation_matrix()
# Rotate true inertial vectors to body frame (what camera sees)
true_vectors_body = (R_true @ true_vectors_inertial.T).T

# Add small angular noise to each vector (simulate measurement noise)
noise_level_deg = 0.01  # small noise in degrees (10 millideg)
noise_level_rad = np.deg2rad(noise_level_deg)
def perturb_unit_vector(v, sigma_rad):
    # rotate vector by a small random rotation whose angle ~ N(0, sigma_rad)
    # generate small axis-angle
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.random.normal(scale=sigma_rad)
    # Rodrigues' rotation
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
    v2 = R @ v
    return v2 / np.linalg.norm(v2)

observed_true_vectors = np.stack([perturb_unit_vector(v, noise_level_rad) for v in true_vectors_body], axis=0)

# Add some false detections (spurious stars)
num_false = 4
false_vectors = ra_dec_to_vector(np.random.uniform(0,360,size=num_false),
                                 np.degrees(np.arcsin(np.random.uniform(-1,1,size=num_false))))
observed_all_vectors = np.vstack([observed_true_vectors, false_vectors])
observed_count = observed_all_vectors.shape[0]

print(f"Simulated observed star count: {observed_count} (true={num_true_visible}, false={num_false})")

# -------------------------
# 4) Build observed pyramid signatures and match against catalog
# -------------------------
from functools import lru_cache

observed_pyramids = []
for comb in itertools.combinations(range(observed_count), 4):
    vecs = observed_all_vectors[list(comb)]
    sig = pyramid_signature_from_vectors(vecs)
    observed_pyramids.append({"indices": comb, "signature": sig})
observed_pyramids = np.array(observed_pyramids, dtype=object)
print(f"Computed {len(observed_pyramids)} observed pyramids.")

# Matching strategy: for each observed pyramid, find nearest catalog pyramid by L2 distance between signatures.
# Use a heuristic tolerance (degrees).
tolerance_deg = 0.05  # allowed mismatch on the 6-angle signature (tuneable)
candidates = []  # list of matches: (observed_pyr_index, catalog_pyr_index, distance)

# For speed, vectorize catalog signatures into an array
catalog_sigs = np.stack([p["signature"] for p in catalog_pyramids], axis=0)  # (N_cat_pyr, 6)

obs_sigs = np.stack([p["signature"] for p in observed_pyramids], axis=0)

# compute pairwise L2 distances (this is O(M*N) but OK for our small sizes)
# We'll find the best catalog match for each observed pyramid
for oi, obs in enumerate(observed_pyramids):
    diffs = catalog_sigs - obs["signature"]  # broadcasting
    dists = np.linalg.norm(diffs, axis=1)
    best_idx = np.argmin(dists)
    best_dist = dists[best_idx]
    if best_dist < tolerance_deg:
        candidates.append((oi, best_idx, float(best_dist)))

print(f"Found {len(candidates)} signature-based candidate matches (tolerance {tolerance_deg} deg).")

# -------------------------
# 5) For each candidate, form correspondences and verify by rotation + inlier counting
# -------------------------
def apply_rotation(R, vecs):
    return (R @ vecs.T).T

best_solution = None
solutions = []

for (oi, ci, dist) in candidates:
    obs_comb = observed_pyramids[oi]["indices"]
    cat_comb = catalog_pyramids[ci]["indices"]
    # Build arrays of matching vectors: observed -> catalog (observed in body frame, catalog in inertial)
    obs_vecs = observed_all_vectors[list(obs_comb)]
    cat_vecs = np.stack([catalog[i]["vec"] for i in cat_comb], axis=0)
    # We want rotation R such that R @ cat_vecs.T ~ obs_vecs.T  -> body = R * inertial
    R_est = kabsch_rotation(cat_vecs, obs_vecs)
    # Apply R_est to all catalog vectors of the visible catalog subset (we will test against observed_all_vectors)
    # But since observed contains false stars, we will check which catalog stars align with any observed vector (within angular threshold)
    transformed_catalog = apply_rotation(R_est, np.stack([c["vec"] for c in catalog], axis=0))
    # For each transformed catalog vector, find nearest observed vector angular distance
    angular_dists = np.zeros((len(catalog), observed_count))
    for i in range(len(catalog)):
        for j in range(observed_count):
            angular_dists[i,j] = angular_distance_deg(transformed_catalog[i], observed_all_vectors[j])
    # Count catalog->observed matches within angular acceptance
    ang_accept_deg = 0.2  # acceptance threshold for a catalog star to match an observed detection
    matches = []
    for i in range(len(catalog)):
        jmin = np.argmin(angular_dists[i])
        dmin = angular_dists[i, jmin]
        if dmin < ang_accept_deg:
            matches.append((i, jmin, dmin))
    # Count inliers that correspond to *true* catalog stars (for diagnostics)
    inlier_count = len(matches)
    # Compute mean residual for matches
    mean_resid = np.mean([m[2] for m in matches]) if matches else None
    solutions.append({
        "observed_pyr_index": oi,
        "catalog_pyr_index": ci,
        "signature_dist": dist,
        "R_est": R_est,
        "inlier_count": inlier_count,
        "mean_residual_deg": mean_resid,
        "matches": matches
    })
    # Track best by inlier_count, then by mean residual
    if best_solution is None or (inlier_count > best_solution["inlier_count"]) or (inlier_count == best_solution["inlier_count"] and (mean_resid is not None and mean_resid < best_solution["mean_residual_deg"])):
        best_solution = solutions[-1]

if best_solution is None:
    print("No valid pyramid matches found under given tolerances.")
else:
    print("Best solution found:")
    print(f" - observed pyramid index: {best_solution['observed_pyr_index']}")
    print(f" - catalog pyramid index: {best_solution['catalog_pyr_index']}")
    print(f" - signature distance: {best_solution['signature_dist']:.6f} deg")
    print(f" - inlier count: {best_solution['inlier_count']}")
    print(f" - mean residual (deg): {best_solution['mean_residual_deg']:.6f}")

    # Show matched correspondences (catalog id -> observed index)
    matched_info = []
    for (cat_i, obs_j, d) in best_solution["matches"]:
        matched_info.append({
            "catalog_id": catalog[cat_i]["id"],
            "catalog_ra": catalog[cat_i]["ra_deg"],
            "catalog_dec": catalog[cat_i]["dec_deg"],
            "observed_index": int(obs_j),
            "angular_residual_deg": float(d)
        })
    import pandas as pd
    df_matches = pd.DataFrame(matched_info)
    # display dataframe to user
    try:
        from caas_jupyter_tools import display_dataframe_to_user
        display_dataframe_to_user("Matched catalog -> observed correspondences", df_matches)
    except Exception:
        print(df_matches.to_string(index=False))

    # Evaluate rotation error compared to ground-truth R_true
    # We estimated R_est mapping inertial->body. R_true is the true inertial->body rotation used.
    R_err = best_solution["R_est"] @ R_true.T
    # angle of rotation represented by R_err
    trace = np.clip(np.trace(R_err), -1.0, 3.0)
    angle_err_rad = np.arccos((trace - 1) / 2.0)
    angle_err_deg = np.degrees(angle_err_rad)
    print(f"Estimated attitude error (deg): {angle_err_deg:.6f}")

# -------------------------
# 6) Example: Use found rotation to identify catalog IDs for observed detections
# -------------------------
if best_solution is not None:
    R_final = best_solution["R_est"]
    transformed_cat = apply_rotation(R_final, np.stack([c["vec"] for c in catalog], axis=0))
    # For each observed vector, find nearest catalog star under the rotation
    observed_to_catalog = []
    for j in range(observed_count):
        dists = [angular_distance_deg(transformed_cat[i], observed_all_vectors[j]) for i in range(len(catalog))]
        imin = int(np.argmin(dists))
        dmin = float(np.min(dists))
        observed_to_catalog.append({
            "observed_index": j,
            "best_catalog_id": catalog[imin]["id"],
            "best_catalog_ra": catalog[imin]["ra_deg"],
            "best_catalog_dec": catalog[imin]["dec_deg"],
            "angular_residual_deg": dmin
        })
    df_map = pd.DataFrame(observed_to_catalog)
    try:
        display_dataframe_to_user("Observed -> Catalog mapping (using best rotation)", df_map)
    except Exception:
        print(df_map.to_string(index=False))

# -------------------------
# Save datasets to disk
# -------------------------
import os
pwd = os.getcwd()
os.makedirs(pwd + "/data/star_pipeline_output", exist_ok=True)
# Save catalog CSV
import csv
catalog_csv_path = pwd + "/data/star_pipeline_output/catalog.csv"
with open(catalog_csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["id","ra_deg","dec_deg","mag","vec_x","vec_y","vec_z"])
    for c in catalog:
        writer.writerow([c["id"], c["ra_deg"], c["dec_deg"], c["mag"], c["vec"][0], c["vec"][1], c["vec"][2]])
# Save observed vectors
np.savetxt(pwd + "/data/star_pipeline_output/observed_vectors.csv", observed_all_vectors, delimiter=",")
# Save R_true and R_est if available
np.save(pwd + "/data/star_pipeline_output/R_true.npy", R_true)
if best_solution is not None:
    np.save(pwd + "/data/star_pipeline_output/R_est.npy", best_solution["R_est"])

print("\nFiles saved to data/star_pipeline_output:")
for p in ["catalog.csv", "observed_vectors.csv", "R_true.npy", "R_est.npy"]:
    full = pwd + "/data/star_pipeline_output/" + p
    if os.path.exists(full):
        print(" -", full)





