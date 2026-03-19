#!/usr/bin/env python3
"""
eval_slam.py
============
Evaluates the full SLAM pipeline against ground truth, now including Landmarks.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# ... [Keep align_timestamps, compute_ate, compute_rpe, compute_sigma_consistency exactly as they were] ...

def align_timestamps(df_query, df_ref, max_dt=0.2):
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
    times = est['timestamp'].values
    errors = []
    for i in range(len(times)):
        target = times[i] + delta_t
        j = np.searchsorted(times, target)
        if j >= len(times): continue
        gt_dx = gt.iloc[j]['x'] - gt.iloc[i]['x']
        gt_dy = gt.iloc[j]['y'] - gt.iloc[i]['y']
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
        'sigma_x': sigma_x, 'sigma_y': sigma_y, 'error_x': dx, 'error_y': dy,
        'consistent_x_pct': float(np.mean(consistent_x) * 100),
        'consistent_y_pct': float(np.mean(consistent_y) * 100),
        'consistent_both_pct': float(np.mean(consistent_both) * 100),
    }

# ─────────────────────────────────────────────
# Updated Plotting
# ─────────────────────────────────────────────

def plot_trajectories(ekf, gt, odom, landmarks, out_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Plot Landmarks first (Background)
    if landmarks is not None and not landmarks.empty:
        # We only want the unique final position of each landmark ID
        # (Otherwise they will look like lines if the EKF updated them)
        final_landmarks = landmarks.sort_values('timestamp').groupby('id').tail(1)
        
        # Split into Active and Archived based on alpha from your EKF code
        active = final_landmarks[final_landmarks['alpha'] > 0.5]
        archived = final_landmarks[final_landmarks['alpha'] <= 0.5]
        
        ax.scatter(archived['x'], archived['y'], c='gray', s=20, alpha=0.3, 
                   marker='o', label='Archived Landmarks', edgecolors='none')
        ax.scatter(active['x'], active['y'], c='orange', s=40, alpha=0.8, 
                   marker='*', label='Active Landmarks', edgecolors='black', linewidths=0.5)

    # 2. Plot Trajectories
    ax.plot(gt['x'], gt['y'], 'g-', linewidth=2, label='Ground Truth', zorder=5)
    if odom is not None:
        ax.plot(odom['x'], odom['y'], 'r--', linewidth=1.5, label='Wheel Odometry', alpha=0.6, zorder=4)
    ax.plot(ekf['x'], ekf['y'], 'b-', linewidth=1.5, label='EKF Estimate', alpha=0.9, zorder=6)

    # Mark start and end
    ax.plot(gt['x'].iloc[0], gt['y'].iloc[0], 'ko', markersize=8, label='Start', zorder=10)
    ax.plot(gt['x'].iloc[-1], gt['y'].iloc[-1], 'g^', markersize=8, label='End (GT)', zorder=10)
    ax.plot(ekf['x'].iloc[-1], ekf['y'].iloc[-1], 'b^', markersize=8, label='End (EKF)', zorder=10)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('SLAM Trajectory & Landmark Map')
    ax.legend(loc='upper right', fontsize='small', framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'trajectories.png'), dpi=150)
    plt.close()
    print("  Saved: trajectories.png (with landmarks)")

# ... [Keep plot_ate_over_time, plot_sigma_consistency, plot_covariance_over_time, save_summary_table as they were] ...

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

def plot_sigma_consistency(times, sigma_data, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t0 = times[0]
    t_rel = times - t0
    for i, (axis_name, err_key, sig_key) in enumerate([('X', 'error_x', 'sigma_x'), ('Y', 'error_y', 'sigma_y')]):
        ax = axes[i]
        err, sig = sigma_data[err_key], sigma_data[sig_key]
        ax.plot(t_rel, err, 'b-', linewidth=1.0, label=f'Error {axis_name}')
        ax.fill_between(t_rel, -sig, sig, alpha=0.2, color='orange', label='3σ bound')
        violations = np.abs(err) > sig
        if np.any(violations):
            ax.fill_between(t_rel, err, 0, where=violations, color='red', alpha=0.4, label='Violations')
        ax.set_ylabel(f'Error {axis_name} [m]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sigma_consistency.png'), dpi=150)
    plt.close()

def plot_covariance_over_time(ekf, times, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    t0 = times[0]
    t_rel = times - t0
    ax.plot(t_rel, np.sqrt(ekf['cov_xx'].values), 'b-', label='σ_x')
    ax.plot(t_rel, np.sqrt(ekf['cov_yy'].values), 'r-', label='σ_y')
    ax.plot(t_rel, np.sqrt(ekf['cov_yaw'].values), 'g-', label='σ_yaw')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('1σ Uncertainty [m or rad]')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'covariance_over_time.png'), dpi=150)
    plt.close()

def save_summary_table(stats, out_dir):
    rows = [{'Metric': k, 'Value': f'{v:.4f}' if isinstance(v, float) else v} for k, v in stats.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'slam_summary.csv'), index=False)

# ─────────────────────────────────────────────
# Updated Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ekf',  required=True, help='ekf_poses.csv')
    parser.add_argument('--gt',   required=True, help='ground_truth.csv')
    parser.add_argument('--landmarks', default=None, help='ekf_landmarks.csv (optional)')
    parser.add_argument('--odom', default=None,  help='wheel_odom.csv (optional)')
    parser.add_argument('--out',  required=True, help='Output directory')
    parser.add_argument('--rpe_window', type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    ekf = load_csv(args.ekf)
    gt  = load_csv(args.gt)
    odom = load_csv(args.odom) if args.odom else None
    lms  = load_csv(args.landmarks) if args.landmarks else None

    # Align timestamps
    print("\nAligning timestamps...")
    ekf_aligned, gt_aligned = align_timestamps(ekf, gt)

    odom_aligned = None
    gt_aligned_odom = None
    if odom is not None:
        odom_aligned, gt_aligned_odom = align_timestamps(odom, gt)

    times = ekf_aligned['timestamp'].values

    # Metric Computations
    ate_errors_ekf, ate_stats = compute_ate(ekf_aligned, gt_aligned)
    ate_errors_odom, ate_stats_odom = (None, {})
    if odom_aligned is not None:
        ate_errors_odom, ate_stats_odom = compute_ate(odom_aligned, gt_aligned_odom)

    _, rpe_stats = compute_rpe(ekf_aligned, gt_aligned, delta_t=args.rpe_window)
    rpe_stats_odom = compute_rpe(odom_aligned, gt_aligned_odom, delta_t=args.rpe_window)[1] if odom_aligned is not None else {}
    sigma_data = compute_sigma_consistency(ekf_aligned, gt_aligned)

    # Compile summary
    summary = {
        'EKF ATE RMSE [m]': ate_stats['ATE_RMSE'],
        'Odometry ATE RMSE [m]': ate_stats_odom.get('ATE_RMSE', np.nan),
        'EKF RPE RMSE [m]': rpe_stats['RPE_RMSE'],
        'End Position Error EKF [m]': float(np.sqrt((ekf_aligned['x'].iloc[-1] - gt_aligned['x'].iloc[-1])**2 + (ekf_aligned['y'].iloc[-1] - gt_aligned['y'].iloc[-1])**2)),
        '3σ Consistent Both [%]': sigma_data['consistent_both_pct'],
        'Landmarks Discovered': len(lms['id'].unique()) if lms is not None else 0,
        'Duration [s]': float(times[-1] - times[0]),
    }

    # Plots
    print("\nGenerating plots...")
    plot_trajectories(ekf_aligned, gt_aligned, odom_aligned, lms, args.out)
    
    times_odom = odom_aligned['timestamp'].values if odom_aligned is not None else None
    plot_ate_over_time(times, ate_errors_ekf, ate_errors_odom, args.out, times_odom=times_odom)
    plot_sigma_consistency(times, sigma_data, args.out)
    plot_covariance_over_time(ekf_aligned, times, args.out)
    save_summary_table(summary, args.out)

if __name__ == '__main__':
    main()