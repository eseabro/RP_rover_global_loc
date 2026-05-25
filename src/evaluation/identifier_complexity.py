#!/usr/bin/env python3
"""
Empirical time-complexity benchmark for identify_geometric.

Generates a synthetic 2D catalog, derives noisy observations from it via a
known similarity transform + Gaussian noise + outliers, then times
identify_geometric across a log-spaced sweep of catalog sizes. Fits a
power law to the medians to estimate the asymptotic exponent.

Usage examples:
    python identifier_complexity.py
    python identifier_complexity.py --N 100 200 500 1000 2000 5000 --trials 7
    python identifier_complexity.py --mode ransac          # skip hash build (pre-built)
    python identifier_complexity.py --mode hash            # only the hash build
"""

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — no DISPLAY needed
import matplotlib.pyplot as plt

# Force line-buffered stdout so progress prints show up immediately even
# when piped/redirected. Equivalent to running with `python -u`.
print = functools.partial(print, flush=True)  # type: ignore[assignment]
try:
    sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
except AttributeError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "ROS_ws" / "src" / "custom_slam"))

print("Importing identifier...")
from custom_slam.identifier import identify_geometric, build_geometric_hash_fast
print("Imports done.")


def make_catalog(n, area=1000.0, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    pts = rng.uniform(0.0, area, size=(n, 2))
    sizes = rng.uniform(0.3, 1.2, size=(n, 2))
    return {"catalog_vectors": pts, "catalog_sizes": sizes}


def make_observations(catalog, m, noise_std=0.05, outlier_frac=0.15,
                      area=1000.0, rng=None):
    """Pick m nearby catalog rocks, apply a random similarity (s=1, random R, t),
    add Gaussian noise, then mix in outliers so RANSAC has real work to do."""
    rng = rng if rng is not None else np.random.default_rng(1)
    cat_pts = catalog["catalog_vectors"]
    cat_sizes = catalog["catalog_sizes"]

    n_inliers = max(int(round(m * (1.0 - outlier_frac))), 3)
    center = rng.uniform(0.0, area, size=2)
    dists = np.linalg.norm(cat_pts - center, axis=1)
    nearest = np.argsort(dists)[:n_inliers]
    inlier_world = cat_pts[nearest]
    inlier_sizes = cat_sizes[nearest]

    theta = rng.uniform(-np.pi, np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    t = rng.uniform(-area, area, size=2)

    # World = s*R*obs + t, with s=1 -> obs = R^T (world - t)
    obs_inliers = (inlier_world - t) @ R
    obs_inliers += rng.normal(0.0, noise_std, size=obs_inliers.shape)

    obs_sizes_in = inlier_sizes + rng.normal(0.0, 0.02, size=inlier_sizes.shape)
    obs_sizes_in = np.clip(obs_sizes_in, 0.1, None)

    n_outliers = m - n_inliers
    if n_outliers > 0:
        extent = float(np.ptp(obs_inliers, axis=0).max()) + 5.0
        out_pts = rng.uniform(-extent, extent, size=(n_outliers, 2))
        out_sizes = rng.uniform(0.3, 1.2, size=(n_outliers, 2))
        obs_pts = np.vstack([obs_inliers, out_pts])
        obs_sizes = np.vstack([obs_sizes_in, out_sizes])
    else:
        obs_pts = obs_inliers
        obs_sizes = obs_sizes_in

    order = rng.permutation(len(obs_pts))
    return {"observed_vectors": obs_pts[order],
            "observed_sizes": obs_sizes[order]}


def time_call(catalog, obs, hash_index, ransac_iters):
    t0 = time.perf_counter()
    identify_geometric(
        obs, catalog,
        hash_index=hash_index,
        ransac_iters=ransac_iters,
        early_exit_fraction=2.0,  # disabled so every call runs all iters
    )
    return time.perf_counter() - t0


def time_hash_build(catalog):
    t0 = time.perf_counter()
    h = build_geometric_hash_fast(catalog["catalog_vectors"],
                                  catalog["catalog_sizes"])
    return time.perf_counter() - t0, h


def fit_power_law(xs, ys):
    """log y = a log x + b -> return (a, b)."""
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    a, b = np.polyfit(lx, ly, 1)
    return float(a), float(b)


def run_sweep(N_values, M, trials, ransac_iters, mode):
    medians = []
    for N in N_values:
        cat_rng = np.random.default_rng(int(N))
        catalog = make_catalog(int(N), rng=cat_rng)

        prebuilt_hash = None
        if mode == "ransac":
            _, prebuilt_hash = time_hash_build(catalog)

        times = []
        for trial in range(trials):
            obs_rng = np.random.default_rng(10_000 + trial)
            obs = make_observations(catalog, M, rng=obs_rng)
            if mode == "hash":
                t, _ = time_hash_build(catalog)
            elif mode == "ransac":
                t = time_call(catalog, obs, prebuilt_hash, ransac_iters)
            else:  # full = hash + ransac
                t = time_call(catalog, obs, None, ransac_iters)
            times.append(t)
        med = float(np.median(times))
        medians.append(med)
        print(f"N={int(N):>6d}  median {med*1e3:9.2f} ms  "
              f"(min {min(times)*1e3:.1f}, max {max(times)*1e3:.1f}, "
              f"trials={trials})")
    return np.array(medians)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, nargs="+",
                   default=[100, 200, 500, 1000, 2000, 5000, 10000],
                   help="Catalog sizes to sweep")
    p.add_argument("--M", type=int, default=25,
                   help="Number of observations per query")
    p.add_argument("--trials", type=int, default=5,
                   help="Trials per N (median is reported)")
    p.add_argument("--ransac-iters", type=int, default=2000)
    p.add_argument("--mode", choices=["full", "ransac", "hash"], default="full",
                   help="full: hash build + RANSAC; ransac: pre-built hash; "
                        "hash: only the hash build")
    p.add_argument("--out", default=str(SCRIPT_DIR / "identifier_complexity.png"))
    args = p.parse_args()

    N_arr = np.array(args.N, dtype=int)
    print(f"Sweeping N={list(N_arr)}, M={args.M}, mode={args.mode}, "
          f"ransac_iters={args.ransac_iters}, trials={args.trials}\n")

    medians = run_sweep(N_arr, args.M, args.trials, args.ransac_iters, args.mode)

    exp, intercept = fit_power_law(N_arr, medians)
    print(f"\nFitted power-law exponent  T(N) ~ N^a   ->   a = {exp:.3f}")
    print(f"Implied scale constant     C = exp({intercept:.3f}) = {np.exp(intercept):.3e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N_arr, medians, "o-", label=f"measured (mode={args.mode})")
    for k, label, style in [(1, "O(N)", "--"), (2, "O(N²)", ":")]:
        ref = medians[0] * (N_arr / N_arr[0]) ** k
        ax.loglog(N_arr, ref, style, alpha=0.5, label=label)
    fit_line = np.exp(intercept) * N_arr.astype(float) ** exp
    ax.loglog(N_arr, fit_line, "-", alpha=0.5,
              label=f"fit ~ N^{exp:.2f}")
    ax.set_xlabel("Catalog size N")
    ax.set_ylabel("Median runtime (s)")
    ax.set_title(f"identify_geometric scaling   (M={args.M}, iters={args.ransac_iters})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
