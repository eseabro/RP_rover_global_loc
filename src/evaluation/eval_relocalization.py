#!/usr/bin/env python3
"""
eval_relocalization.py
======================
Evaluates the sequential relocalization performance.

Metrics:
  - Distance to convergence: how far the rover travels before first
    successful global localization
  - Localization success rate vs distance travelled
  - Global pose error vs ground truth at each relocalization event
  - Time between relocalization events

Usage:
    python3 eval_relocalization.py --global global_poses.csv
                                   --gt ground_truth.csv
                                   --odom wheel_odom.csv
                                   --out results/reloc/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    return df.sort_values('timestamp').reset_index(drop=True)


def align_timestamps(df_query, df_ref, max_dt=0.5):
    ref_times = df_ref['timestamp'].values
    tree = KDTree(ref_times.reshape(-1, 1))
    aligned_query, aligned_ref = [], []
    for _, row in df_query.iterrows():
        dist, idx = tree.query([[row['timestamp']]])
        if dist[0] < max_dt:
            aligned_query.append(row)
            aligned_ref.append(df_ref.iloc[idx[0]])
    return (pd.DataFrame(aligned_query).reset_index(drop=True),
            pd.DataFrame(aligned_ref).reset_index(drop=True))


def compute_distance_travelled(odom):
    """Cumulative distance travelled from odometry."""
    dx = np.diff(odom['x'].values)
    dy = np.diff(odom['y'].values)
    step_dist = np.sqrt(dx**2 + dy**2)
    return np.concatenate([[0.0], np.cumsum(step_dist)])


def get_distance_at_time(odom, cum_dist, query_time):
    """Interpolate cumulative distance at a given timestamp."""
    times = odom['timestamp'].values
    idx = np.searchsorted(times, query_time)
    idx = np.clip(idx, 0, len(cum_dist) - 1)
    return cum_dist[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--global_poses', required=True, help='global_poses.csv')
    parser.add_argument('--gt',           required=True, help='ground_truth.csv')
    parser.add_argument('--odom',         required=True, help='wheel_odom.csv')
    parser.add_argument('--out',          required=True, help='Output directory')
    parser.add_argument('--error_threshold', type=float, default=2.0,
                        help='Max error [m] to count as successful localization (default: 2.0)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    global_poses = load_csv(args.global_poses)
    gt           = load_csv(args.gt)
    odom         = load_csv(args.odom)

    print(f"  Global pose events: {len(global_poses)}")
    print(f"  Ground truth:       {len(gt)} samples")
    print(f"  Wheel odometry:     {len(odom)} samples")

    if len(global_poses) == 0:
        print("  ERROR: No global pose events found. Matcher may not have triggered.")
        return

    # Cumulative distance
    cum_dist = compute_distance_travelled(odom)
    total_distance = cum_dist[-1]
    print(f"  Total distance travelled: {total_distance:.2f} m")

    # Align global poses with ground truth
    gp_aligned, gt_aligned = align_timestamps(global_poses, gt)

    # Compute error at each relocalization event
    dx = gp_aligned['x'].values - gt_aligned['x'].values
    dy = gp_aligned['y'].values - gt_aligned['y'].values
    errors = np.sqrt(dx**2 + dy**2)

    # Distance at each event
    event_distances = np.array([
        get_distance_at_time(odom, cum_dist, t)
        for t in gp_aligned['timestamp'].values
    ])

    # ── Distance to convergence ──────────────────────────
    successful = errors < args.error_threshold
    if np.any(successful):
        first_success_idx = np.argmax(successful)
        dist_to_convergence = event_distances[first_success_idx]
        time_to_convergence = gp_aligned['timestamp'].iloc[first_success_idx] - \
                              global_poses['timestamp'].iloc[0]
    else:
        dist_to_convergence = float('nan')
        time_to_convergence = float('nan')

    # ── Success rate vs distance bins ────────────────────
    bin_edges = np.arange(0, total_distance + 5.0, 5.0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    success_rates = []
    for i in range(len(bin_edges) - 1):
        mask = (event_distances >= bin_edges[i]) & (event_distances < bin_edges[i+1])
        if np.sum(mask) == 0:
            success_rates.append(float('nan'))
        else:
            success_rates.append(float(np.mean(successful[mask]) * 100))

    # ── Inter-event timing ───────────────────────────────
    event_times = gp_aligned['timestamp'].values
    if len(event_times) > 1:
        inter_event_dt = np.diff(event_times)
        mean_reloc_freq = 1.0 / np.mean(inter_event_dt) if np.mean(inter_event_dt) > 0 else 0
    else:
        inter_event_dt = np.array([])
        mean_reloc_freq = 0.0

    # ── Summary ──────────────────────────────────────────
    summary = {
        'Total relocalization events':    len(gp_aligned),
        'Successful events':              int(np.sum(successful)),
        'Overall success rate [%]':       float(np.mean(successful) * 100),
        'Distance to convergence [m]':    dist_to_convergence,
        'Time to convergence [s]':        time_to_convergence,
        'Mean reloc error [m]':           float(np.mean(errors)),
        'Std reloc error [m]':            float(np.std(errors)),
        'Max reloc error [m]':            float(np.max(errors)),
        'Min reloc error [m]':            float(np.min(errors)),
        'Mean inter-event time [s]':      float(np.mean(inter_event_dt)) if len(inter_event_dt) > 0 else float('nan'),
        'Mean relocalization rate [Hz]':  mean_reloc_freq,
        'Total distance [m]':             total_distance,
        'Error threshold used [m]':       args.error_threshold,
    }

    print("\n" + "="*55)
    print("  SEQUENTIAL RELOCALIZATION EVALUATION SUMMARY")
    print("="*55)
    for k, v in summary.items():
        val_str = f'{v:.4f}' if isinstance(v, float) else str(v)
        print(f"  {k:<40} {val_str}")
    print("="*55)

    # Save summary CSV
    rows = [{'Metric': k, 'Value': f'{v:.4f}' if isinstance(v, float) else v}
            for k, v in summary.items()]
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'reloc_summary.csv'), index=False)

    # ── Plots ─────────────────────────────────────────────

    # 1. Error at each relocalization event
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['green' if s else 'red' for s in successful]
    ax.scatter(event_distances, errors, c=colors, s=40, zorder=3)
    ax.axhline(args.error_threshold, color='orange', linestyle='--',
               label=f'Success threshold ({args.error_threshold}m)')
    if not np.isnan(dist_to_convergence):
        ax.axvline(dist_to_convergence, color='blue', linestyle=':',
                   label=f'Convergence at {dist_to_convergence:.1f}m')
    green_patch = mpatches.Patch(color='green', label='Successful')
    red_patch   = mpatches.Patch(color='red',   label='Failed')
    ax.legend(handles=[green_patch, red_patch,
                        plt.Line2D([0],[0], color='orange', linestyle='--'),
                        plt.Line2D([0],[0], color='blue',   linestyle=':')])
    ax.set_xlabel('Distance Travelled [m]')
    ax.set_ylabel('Localization Error [m]')
    ax.set_title('Global Localization Error vs Distance Travelled')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'reloc_error_vs_distance.png'), dpi=150)
    plt.close()
    print("  Saved: reloc_error_vs_distance.png")

    # 2. Success rate vs distance bins
    valid = ~np.isnan(success_rates)
    if np.any(valid):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(bin_centers[valid], np.array(success_rates)[valid],
               width=4.0, color='steelblue', edgecolor='white', alpha=0.85)
        ax.set_xlabel('Distance Travelled [m]')
        ax.set_ylabel('Success Rate [%]')
        ax.set_title('Localization Success Rate vs Distance Travelled')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'success_rate_vs_distance.png'), dpi=150)
        plt.close()
        print("  Saved: success_rate_vs_distance.png")

    # 3. Trajectory with relocalization events marked
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gt['x'], gt['y'], 'g-', linewidth=1.5, label='Ground Truth', alpha=0.7)
    ax.scatter(gt_aligned['x'][successful], gt_aligned['y'][successful],
               c='green', s=60, zorder=4, label='Successful reloc')
    ax.scatter(gt_aligned['x'][~successful], gt_aligned['y'][~successful],
               c='red', s=60, zorder=4, label='Failed reloc', marker='x')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Relocalization Events on Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'reloc_on_trajectory.png'), dpi=150)
    plt.close()
    print("  Saved: reloc_on_trajectory.png")

    print(f"\n  All results saved to: {args.out}")


import matplotlib.patches as mpatches

if __name__ == '__main__':
    main()
