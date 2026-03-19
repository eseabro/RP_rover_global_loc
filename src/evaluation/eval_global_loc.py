#!/usr/bin/env python3
"""
eval_global_loc.py
==================
Evaluates the global localization algorithm (constellation matching).

Metrics:
  - Mean error and error variance across all localization events
  - Number of inliers per event (requires matcher_node to publish stats)
  - Query time vs database size (requires matcher_node to publish stats)

NOTE: Some metrics require matcher_node.py to be updated to publish
      a /matcher/stats topic. A template for that is included below.

Usage:
    python3 eval_global_loc.py --global global_poses.csv
                               --gt ground_truth.csv
                               --stats matcher_stats.csv  (optional)
                               --out results/global_loc/
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
    return pd.read_csv(path).sort_values('timestamp').reset_index(drop=True)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--global_poses', required=True, help='global_poses.csv')
    parser.add_argument('--gt',           required=True, help='ground_truth.csv')
    parser.add_argument('--stats',        default=None,
                        help='matcher_stats.csv (inliers, query_time) - optional')
    parser.add_argument('--out',          required=True, help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    global_poses = load_csv(args.global_poses)
    gt = load_csv(args.gt)

    print(f"  Global pose events: {len(global_poses)}")
    print(f"  Ground truth:       {len(gt)} samples")

    if len(global_poses) == 0:
        print("  ERROR: No global pose events found.")
        return

    # Align
    gp_aligned, gt_aligned = align_timestamps(global_poses, gt)

    dx = gp_aligned['x'].values - gt_aligned['x'].values
    dy = gp_aligned['y'].values - gt_aligned['y'].values
    errors = np.sqrt(dx**2 + dy**2)
    yaw_errors = np.abs(gp_aligned['yaw'].values - gt_aligned['yaw'].values)
    yaw_errors = np.minimum(yaw_errors, 2*np.pi - yaw_errors)  # wrap to [0, pi]

    # ── Core metrics ─────────────────────────────────────
    summary = {
        'Total localization events':     len(gp_aligned),
        'Mean position error [m]':       float(np.mean(errors)),
        'Std position error [m]':        float(np.std(errors)),
        'Median position error [m]':     float(np.median(errors)),
        'Max position error [m]':        float(np.max(errors)),
        'Min position error [m]':        float(np.min(errors)),
        'Mean X error [m]':              float(np.mean(dx)),
        'Mean Y error [m]':              float(np.mean(dy)),
        'Std X error [m]':               float(np.std(dx)),
        'Std Y error [m]':               float(np.std(dy)),
        'Mean yaw error [deg]':          float(np.degrees(np.mean(yaw_errors))),
        'Std yaw error [deg]':           float(np.degrees(np.std(yaw_errors))),
        'Events with error < 0.5m [%]':  float(np.mean(errors < 0.5) * 100),
        'Events with error < 1.0m [%]':  float(np.mean(errors < 1.0) * 100),
        'Events with error < 2.0m [%]':  float(np.mean(errors < 2.0) * 100),
    }

    # ── Matcher stats (inliers, query time) ──────────────
    has_stats = False
    if args.stats and os.path.exists(args.stats):
        stats_df = load_csv(args.stats)
        has_stats = True
        print(f"  Matcher stats:      {len(stats_df)} records")

        if 'n_inliers' in stats_df.columns:
            summary['Mean inliers per query'] = float(stats_df['n_inliers'].mean())
            summary['Std inliers per query']  = float(stats_df['n_inliers'].std())
            summary['Min inliers']            = float(stats_df['n_inliers'].min())
            summary['Max inliers']            = float(stats_df['n_inliers'].max())

        if 'n_outliers' in stats_df.columns:
            total = stats_df['n_inliers'] + stats_df['n_outliers']
            inlier_ratio = stats_df['n_inliers'] / total.replace(0, np.nan)
            summary['Mean inlier ratio [%]'] = float(inlier_ratio.mean() * 100)

        if 'query_time_ms' in stats_df.columns:
            summary['Mean query time [ms]'] = float(stats_df['query_time_ms'].mean())
            summary['Max query time [ms]']  = float(stats_df['query_time_ms'].max())

        if 'db_size' in stats_df.columns:
            summary['Database size (rocks)'] = int(stats_df['db_size'].iloc[-1])
    else:
        print("  NOTE: No matcher stats CSV provided.")
        print("        To get inlier/query_time metrics, add this publisher to matcher_node.py:")
        print("        See MATCHER_NODE_STATS_TEMPLATE below.")

    # ── Print summary ─────────────────────────────────────
    print("\n" + "="*55)
    print("  GLOBAL LOCALIZATION EVALUATION SUMMARY")
    print("="*55)
    for k, v in summary.items():
        val_str = f'{v:.4f}' if isinstance(v, float) else str(v)
        print(f"  {k:<40} {val_str}")
    print("="*55)

    rows = [{'Metric': k, 'Value': f'{v:.4f}' if isinstance(v, float) else v}
            for k, v in summary.items()]
    pd.DataFrame(rows).to_csv(
        os.path.join(args.out, 'global_loc_summary.csv'), index=False)

    # ── Plots ─────────────────────────────────────────────

    # 1. Error distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(errors, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(np.mean(errors), color='red', linestyle='--',
                    label=f'Mean: {np.mean(errors):.3f}m')
    axes[0].axvline(np.median(errors), color='orange', linestyle=':',
                    label=f'Median: {np.median(errors):.3f}m')
    axes[0].set_xlabel('Position Error [m]')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Global Localization Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(np.degrees(yaw_errors), bins=20,
                 color='coral', edgecolor='white', alpha=0.85)
    axes[1].axvline(np.degrees(np.mean(yaw_errors)), color='red', linestyle='--',
                    label=f'Mean: {np.degrees(np.mean(yaw_errors)):.2f}°')
    axes[1].set_xlabel('Yaw Error [deg]')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Global Localization Yaw Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'error_distribution.png'), dpi=150)
    plt.close()
    print("  Saved: error_distribution.png")

    # 2. X/Y error scatter (bias visualization)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(dx, dy, alpha=0.6, s=40, color='steelblue')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('X Error [m]')
    ax.set_ylabel('Y Error [m]')
    ax.set_title('Error Scatter (Bias Visualization)')
    # Draw 1σ ellipse
    cov_mat = np.cov(dx, dy)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 1][::-1]))
    from matplotlib.patches import Ellipse
    for n_sigma in [1, 2, 3]:
        ell = Ellipse(xy=(np.mean(dx), np.mean(dy)),
                      width=2*n_sigma*np.sqrt(eigenvalues[0]),
                      height=2*n_sigma*np.sqrt(eigenvalues[1]),
                      angle=angle,
                      edgecolor='red', facecolor='none',
                      linewidth=1.5, linestyle='--',
                      label=f'{n_sigma}σ bound' if n_sigma == 1 else f'_{n_sigma}σ')
        ax.add_patch(ell)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'error_scatter.png'), dpi=150)
    plt.close()
    print("  Saved: error_scatter.png")

    # 3. Inliers plot (if stats available)
    if has_stats and 'n_inliers' in stats_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Inlier count over time
        t0 = stats_df['timestamp'].iloc[0]
        axes[0].plot(stats_df['timestamp'] - t0, stats_df['n_inliers'],
                     'b-o', markersize=4)
        axes[0].set_xlabel('Time [s]')
        axes[0].set_ylabel('Number of Inliers')
        axes[0].set_title('Inlier Count over Time')
        axes[0].grid(True, alpha=0.3)

        # Query time if available
        if 'query_time_ms' in stats_df.columns:
            axes[1].scatter(stats_df['n_inliers'], stats_df['query_time_ms'],
                            alpha=0.6, s=40, color='coral')
            axes[1].set_xlabel('Number of Inliers')
            axes[1].set_ylabel('Query Time [ms]')
            axes[1].set_title('Query Time vs Inlier Count')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'inlier_stats.png'), dpi=150)
        plt.close()
        print("  Saved: inlier_stats.png")

    print(f"\n  All results saved to: {args.out}")
    print_matcher_node_template()


def print_matcher_node_template():
    """Prints the code to add to matcher_node.py to publish stats."""
    print("""
─────────────────────────────────────────────────────────────
  TO ENABLE INLIER / QUERY TIME METRICS, add this to matcher_node.py:
─────────────────────────────────────────────────────────────

  # In __init__:
  from std_msgs.msg import Float32MultiArray
  self.stats_pub = self.create_publisher(Float32MultiArray, '/matcher/stats', 10)

  # After each match attempt, publish:
  import time
  t_start = time.time()
  # ... your matching code ...
  query_time_ms = (time.time() - t_start) * 1000

  stats_msg = Float32MultiArray()
  stats_msg.data = [
      float(n_inliers),
      float(n_outliers),
      float(query_time_ms),
      float(db_size)
  ]
  self.stats_pub.publish(stats_msg)

  # Then in extract_bag.py, add a matcher_stats extractor for /matcher/stats
─────────────────────────────────────────────────────────────
""")


if __name__ == '__main__':
    main()
