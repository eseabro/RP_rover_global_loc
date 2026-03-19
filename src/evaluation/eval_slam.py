#!/usr/bin/env python3
"""
eval_slam.py
============
Evaluates the full SLAM pipeline against ground truth.

Metrics:
  - Absolute Trajectory Error (ATE) — RMSE of position error over full trajectory
  - Relative Pose Error (RPE) — local drift over fixed time windows
  - 3-sigma consistency check — error must stay within 3*sqrt(covariance)
  - End-of-drive position error and covariance
  - Lat/Lon equivalent error (metres, since we work in local ENU frame)

Usage:
    python3 eval_slam.py --ekf ekf_poses.csv --gt ground_truth.csv
                         --odom wheel_odom.csv --out results/slam/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import KDTree


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def align_timestamps(df_query, df_ref, max_dt=0.2):
    """
    For each row in df_query, find the nearest timestamp in df_ref.
    Returns aligned (query, ref) DataFrames with same length.
    """
    ref_times = df_ref['timestamp'].values
    tree = KDTree(ref_times.reshape(-1, 1))

    aligned_query = []
    aligned_ref = []

    for _, row in df_query.iterrows():
        dist, idx = tree.query([[row['timestamp']]])
        if dist[0] < max_dt:
            aligned_query.append(row)
            aligned_ref.append(df_ref.iloc[idx[0]])

    return pd.DataFrame(aligned_query).reset_index(drop=True), \
           pd.DataFrame(aligned_ref).reset_index(drop=True)


def compute_ate(est, gt):
    """
    Absolute Trajectory Error (ATE).
    Returns per-sample errors and summary stats.
    """
    dx = est['x'].values - gt['x'].values
    dy = est['y'].values - gt['y'].values
    errors = np.sqrt(dx**2 + dy**2)
    return errors, {
        'ATE_RMSE':  float(np.sqrt(np.mean(errors**2))),
        'ATE_mean':  float(np.mean(errors)),
        'ATE_max':   float(np.max(errors)),
        'ATE_std':   float(np.std(errors)),
        'ATE_final': float(errors[-1]) if len(errors) > 0 else float('nan')
    }


def compute_rpe(est, gt, delta_t=5.0):
    """
    Relative Pose Error over time windows of delta_t seconds.
    Measures local drift, independent of global alignment.
    """
    times = est['timestamp'].values
    errors = []

    for i in range(len(times)):
        # Find j such that times[j] - times[i] ~ delta_t
        target = times[i] + delta_t
        j = np.searchsorted(times, target)
        if j >= len(times):
            continue

        # Relative translation in ground truth
        gt_dx = gt.iloc[j]['x'] - gt.iloc[i]['x']
        gt_dy = gt.iloc[j]['y'] - gt.iloc[i]['y']

        # Relative translation in estimate
        est_dx = est.iloc[j]['x'] - est.iloc[i]['x']
        est_dy = est.iloc[j]['y'] - est.iloc[i]['y']

        err = np.sqrt((est_dx - gt_dx)**2 + (est_dy - gt_dy)**2)
        errors.append(err)

    errors = np.array(errors)
    return errors, {
        'RPE_RMSE': float(np.sqrt(np.mean(errors**2))) if len(errors) > 0 else float('nan'),
        'RPE_mean': float(np.mean(errors)) if len(errors) > 0 else float('nan'),
        'RPE_max':  float(np.max(errors)) if len(errors) > 0 else float('nan'),
    }


def compute_sigma_consistency(est, gt):
    """
    3-sigma consistency check.
    The position error should remain within 3*sqrt(covariance) at all times.
    Returns fraction of time the estimator is consistent.
    """
    dx = est['x'].values - gt['x'].values
    dy = est['y'].values - gt['y'].values
    error_x = np.abs(dx)
    error_y = np.abs(dy)

    sigma_x = 3.0 * np.sqrt(np.abs(est['cov_xx'].values))
    sigma_y = 3.0 * np.sqrt(np.abs(est['cov_yy'].values))

    consistent_x = error_x < sigma_x
    consistent_y = error_y < sigma_y
    consistent_both = consistent_x & consistent_y

    return {
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'error_x': dx,
        'error_y': dy,
        'consistent_x_pct': float(np.mean(consistent_x) * 100),
        'consistent_y_pct': float(np.mean(consistent_y) * 100),
        'consistent_both_pct': float(np.mean(consistent_both) * 100),
    }


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def plot_trajectories(ekf, gt, odom, out_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gt['x'], gt['y'], 'g-', linewidth=2, label='Ground Truth', zorder=3)
    if odom is not None:
        ax.plot(odom['x'], odom['y'], 'r--', linewidth=1.5, label='Wheel Odometry', alpha=0.8)
    ax.plot(ekf['x'], ekf['y'], 'b-', linewidth=1.5, label='EKF Estimate', alpha=0.9)

    # Mark start and end
    ax.plot(gt['x'].iloc[0], gt['y'].iloc[0], 'go', markersize=10, label='Start')
    ax.plot(gt['x'].iloc[-1], gt['y'].iloc[-1], 'g^', markersize=10, label='End (GT)')
    ax.plot(ekf['x'].iloc[-1], ekf['y'].iloc[-1], 'b^', markersize=10, label='End (EKF)')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'trajectories.png'), dpi=150)
    plt.close()
    print("  Saved: trajectories.png")


def plot_ate_over_time(times, ate_errors_ekf, ate_errors_odom, out_dir, times_odom=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    t0 = times[0]
    t_rel = times - t0

    ax.plot(t_rel, ate_errors_ekf, 'b-', linewidth=1.5, label='EKF ATE')
    if ate_errors_odom is not None:
        t_odom = times_odom if times_odom is not None else times
        t_odom_rel = t_odom - t_odom[0]
        ax.plot(t_odom_rel, ate_errors_odom, 'r--', linewidth=1.5, label='Odometry ATE', alpha=0.8)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Error [m]')
    ax.set_title('Absolute Trajectory Error over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ate_over_time.png'), dpi=150)
    plt.close()
    print("  Saved: ate_over_time.png")


def plot_sigma_consistency(times, sigma_data, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t0 = times[0]
    t_rel = times - t0

    for i, (axis_name, err_key, sig_key) in enumerate([('X', 'error_x', 'sigma_x'),
                                                         ('Y', 'error_y', 'sigma_y')]):
        ax = axes[i]
        err = sigma_data[err_key]
        sig = sigma_data[sig_key]

        ax.plot(t_rel, err, 'b-', linewidth=1.0, label=f'Error {axis_name}')
        ax.fill_between(t_rel, -sig, sig, alpha=0.2, color='orange', label='3σ bound')
        ax.plot(t_rel, sig, 'r--', linewidth=0.8, alpha=0.7)
        ax.plot(t_rel, -sig, 'r--', linewidth=0.8, alpha=0.7)

        # Highlight violations
        violations = np.abs(err) > sig
        if np.any(violations):
            ax.fill_between(t_rel, err, 0,
                            where=violations, color='red', alpha=0.4, label='Violations')

        ax.set_ylabel(f'Error {axis_name} [m]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel('Time [s]')
    fig.suptitle('3σ Consistency Check (Error vs Covariance Bound)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sigma_consistency.png'), dpi=150)
    plt.close()
    print("  Saved: sigma_consistency.png")


def plot_covariance_over_time(ekf, times, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    t0 = times[0]
    t_rel = times - t0

    ax.plot(t_rel, np.sqrt(ekf['cov_xx'].values), 'b-', label='σ_x')
    ax.plot(t_rel, np.sqrt(ekf['cov_yy'].values), 'r-', label='σ_y')
    ax.plot(t_rel, np.sqrt(ekf['cov_yaw'].values), 'g-', label='σ_yaw')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('1σ Uncertainty [m or rad]')
    ax.set_title('EKF Uncertainty (Standard Deviation) over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'covariance_over_time.png'), dpi=150)
    plt.close()
    print("  Saved: covariance_over_time.png")


def save_summary_table(stats, out_dir):
    rows = []
    for key, val in stats.items():
        rows.append({'Metric': key, 'Value': f'{val:.4f}' if isinstance(val, float) else val})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'slam_summary.csv')
    df.to_csv(csv_path, index=False)

    # Also print nicely
    print("\n" + "="*50)
    print("  SLAM PIPELINE EVALUATION SUMMARY")
    print("="*50)
    for _, row in df.iterrows():
        print(f"  {row['Metric']:<35} {row['Value']}")
    print("="*50)
    print(f"\n  Full table saved to: {csv_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ekf',  required=True, help='ekf_poses.csv')
    parser.add_argument('--gt',   required=True, help='ground_truth.csv')
    parser.add_argument('--odom', default=None,  help='wheel_odom.csv (optional)')
    parser.add_argument('--out',  required=True, help='Output directory')
    parser.add_argument('--rpe_window', type=float, default=5.0,
                        help='RPE time window in seconds (default: 5.0)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    ekf = load_csv(args.ekf)
    gt  = load_csv(args.gt)
    odom = load_csv(args.odom) if args.odom else None

    print(f"  EKF poses:     {len(ekf)} samples")
    print(f"  Ground truth:  {len(gt)} samples")
    if odom is not None:
        print(f"  Wheel odom:    {len(odom)} samples")

    # Align timestamps
    print("\nAligning timestamps...")
    ekf_aligned, gt_aligned = align_timestamps(ekf, gt)
    print(f"  Aligned pairs: {len(ekf_aligned)}")

    odom_aligned = None
    gt_aligned_odom = None
    if odom is not None:
        odom_aligned, gt_aligned_odom = align_timestamps(odom, gt)

    times = ekf_aligned['timestamp'].values

    # ── ATE ──────────────────────────────────
    print("\nComputing ATE...")
    ate_errors_ekf, ate_stats = compute_ate(ekf_aligned, gt_aligned)

    ate_errors_odom = None
    ate_stats_odom = {}
    if odom_aligned is not None:
        ate_errors_odom, ate_stats_odom = compute_ate(odom_aligned, gt_aligned_odom)

    # ── RPE ──────────────────────────────────
    print("Computing RPE...")
    _, rpe_stats = compute_rpe(ekf_aligned, gt_aligned, delta_t=args.rpe_window)

    rpe_stats_odom = {}
    if odom_aligned is not None:
        _, rpe_stats_odom = compute_rpe(odom_aligned, gt_aligned_odom, delta_t=args.rpe_window)

    # ── 3-sigma ───────────────────────────────
    print("Computing 3σ consistency...")
    sigma_data = compute_sigma_consistency(ekf_aligned, gt_aligned)

    # ── End-of-drive stats ────────────────────
    end_pos_error = float(np.sqrt(
        (ekf_aligned['x'].iloc[-1] - gt_aligned['x'].iloc[-1])**2 +
        (ekf_aligned['y'].iloc[-1] - gt_aligned['y'].iloc[-1])**2
    ))
    end_sigma_x = float(np.sqrt(abs(ekf_aligned['cov_xx'].iloc[-1])))
    end_sigma_y = float(np.sqrt(abs(ekf_aligned['cov_yy'].iloc[-1])))
    end_sigma_yaw = float(np.sqrt(abs(ekf_aligned['cov_yaw'].iloc[-1])))

    odom_end_error = float('nan')
    if odom_aligned is not None and len(odom_aligned) > 0:
        odom_end_error = float(np.sqrt(
            (odom_aligned['x'].iloc[-1] - gt_aligned_odom['x'].iloc[-1])**2 +
            (odom_aligned['y'].iloc[-1] - gt_aligned_odom['y'].iloc[-1])**2
        ))

    # ── Compile summary ───────────────────────
    summary = {
        # ATE
        'EKF ATE RMSE [m]':           ate_stats['ATE_RMSE'],
        'EKF ATE Mean [m]':            ate_stats['ATE_mean'],
        'EKF ATE Max [m]':             ate_stats['ATE_max'],
        'EKF ATE Std [m]':             ate_stats['ATE_std'],
        'Odometry ATE RMSE [m]':       ate_stats_odom.get('ATE_RMSE', float('nan')),
        'ATE Improvement vs Odom [%]': float(
            (1 - ate_stats['ATE_RMSE'] / ate_stats_odom['ATE_RMSE']) * 100
        ) if ate_stats_odom.get('ATE_RMSE', 0) > 0 else float('nan'),
        # RPE
        'EKF RPE RMSE [m]':            rpe_stats['RPE_RMSE'],
        'EKF RPE Mean [m]':            rpe_stats['RPE_mean'],
        'Odometry RPE RMSE [m]':       rpe_stats_odom.get('RPE_RMSE', float('nan')),
        # End-of-drive
        'End Position Error EKF [m]':  end_pos_error,
        'End Position Error Odom [m]': odom_end_error,
        'End σ_x [m]':                 end_sigma_x,
        'End σ_y [m]':                 end_sigma_y,
        'End σ_yaw [rad]':             end_sigma_yaw,
        # 3-sigma
        '3σ Consistent X [%]':         sigma_data['consistent_x_pct'],
        '3σ Consistent Y [%]':         sigma_data['consistent_y_pct'],
        '3σ Consistent Both [%]':      sigma_data['consistent_both_pct'],
        # Trajectory length
        'Total samples':               len(ekf_aligned),
        'Duration [s]':                float(times[-1] - times[0]),
    }

    # ── Plots ─────────────────────────────────
    print("\nGenerating plots...")
    plot_trajectories(ekf_aligned, gt_aligned, odom_aligned, args.out)
    times_odom = odom_aligned['timestamp'].values if odom_aligned is not None else None
    plot_ate_over_time(times, ate_errors_ekf, ate_errors_odom, args.out, times_odom=times_odom)
    plot_sigma_consistency(times, sigma_data, args.out)
    plot_covariance_over_time(ekf_aligned, times, args.out)

    # ── Save summary ──────────────────────────
    save_summary_table(summary, args.out)


if __name__ == '__main__':
    main()