#!/usr/bin/env python3
"""
plot_trajectory.py
==================
Generates a publication-quality three-path trajectory comparison plot
for the EKF-SLAM thesis section.

Reads CSV files produced by eval_slam.py / extract_bag.py and produces:
  1. trajectory_comparison.png  - XY paths: GT / Wheel Odom / EKF
  2. ate_comparison.png         - ATE over time for both odom and EKF

Usage:
    python3 plot_trajectory.py \
        --csv_dir /home/ws/results/run04_m/csv \
        --out     /home/ws/results/figures/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_csv(csv_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(csv_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def align_to_origin(df: pd.DataFrame,
                    xcol='x', ycol='y') -> pd.DataFrame:
    """Shift trajectory so it starts at (0, 0)."""
    df = df.copy()
    df[xcol] -= df[xcol].iloc[0]
    df[ycol] -= df[ycol].iloc[0]
    return df


def compute_ate(est_x, est_y, gt_x, gt_y) -> np.ndarray:
    """Per-sample ATE (2D Euclidean distance after time alignment)."""
    return np.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2)


def interpolate_to(source_t, source_x, source_y,
                   target_t) -> tuple:
    """Interpolate source trajectory to target timestamps."""
    xi = np.interp(target_t, source_t, source_x)
    yi = np.interp(target_t, source_t, source_y)
    return xi, yi


# ─────────────────────────────────────────────────────────────
# Plot 1: Three-path XY trajectory
# ─────────────────────────────────────────────────────────────

def plot_trajectory(gt: pd.DataFrame,
                    odom: pd.DataFrame,
                    ekf: pd.DataFrame,
                    out_dir: str):

    # Align all paths to start at origin
    gt   = align_to_origin(gt,   'x', 'y')
    odom = align_to_origin(odom, 'x', 'y')
    ekf  = align_to_origin(ekf,  'x', 'y')

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')

    # ── Draw paths ──────────────────────────────────────────
    ax.plot(gt['x'],   gt['y'],
            color='#2ca02c', linewidth=2.0, linestyle='-',
            label='Ground Truth', zorder=3)

    ax.plot(odom['x'], odom['y'],
            color='#d62728', linewidth=1.5, linestyle='--',
            alpha=0.8, label='Wheel Odometry', zorder=2)

    ax.plot(ekf['x'],  ekf['y'],
            color='#1f77b4', linewidth=2.0, linestyle='-',
            alpha=0.9, label='EKF-SLAM', zorder=4)

    # ── Start / End markers ─────────────────────────────────
    for df, color in [(gt, '#2ca02c'), (odom, '#d62728'), (ekf, '#1f77b4')]:
        ax.plot(df['x'].iloc[0],  df['y'].iloc[0],
                'o', color=color, markersize=8, zorder=5)
        ax.plot(df['x'].iloc[-1], df['y'].iloc[-1],
                's', color=color, markersize=8, zorder=5)

    # Start / End labels (only once, on GT)
    ax.annotate('Start',
                xy=(gt['x'].iloc[0], gt['y'].iloc[0]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, color='#333333',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.7, edgecolor='none'))
    ax.annotate('End',
                xy=(gt['x'].iloc[-1], gt['y'].iloc[-1]),
                xytext=(8, -14), textcoords='offset points',
                fontsize=9, color='#333333',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.7, edgecolor='none'))

    # ── End-point error annotation ──────────────────────────
    ekf_end_err  = np.hypot(ekf['x'].iloc[-1]  - gt['x'].iloc[-1],
                             ekf['y'].iloc[-1]  - gt['y'].iloc[-1])
    odom_end_err = np.hypot(odom['x'].iloc[-1] - gt['x'].iloc[-1],
                             odom['y'].iloc[-1] - gt['y'].iloc[-1])

    ax.annotate(f'EKF end error: {ekf_end_err:.2f} m',
                xy=(ekf['x'].iloc[-1], ekf['y'].iloc[-1]),
                xytext=(-80, 15), textcoords='offset points',
                fontsize=8.5, color='#1f77b4',
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#e8f0fa', alpha=0.85, edgecolor='#1f77b4'))

    ax.annotate(f'Odom end error: {odom_end_err:.2f} m',
                xy=(odom['x'].iloc[-1], odom['y'].iloc[-1]),
                xytext=(10, -20), textcoords='offset points',
                fontsize=8.5, color='#d62728',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#fae8e8', alpha=0.85, edgecolor='#d62728'))

    # ── Formatting ──────────────────────────────────────────
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title('Rover Trajectory Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best',
              framealpha=0.9, edgecolor='#cccccc')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.4, linewidth=0.6)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    path = os.path.join(out_dir, 'trajectory_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Saved: trajectory_comparison.png")


# ─────────────────────────────────────────────────────────────
# Plot 2: ATE over time
# ─────────────────────────────────────────────────────────────

def plot_ate(gt: pd.DataFrame,
             odom: pd.DataFrame,
             ekf: pd.DataFrame,
             out_dir: str):

    # Time axis: use EKF timestamps as reference
    t0 = gt['timestamp'].iloc[0]
    gt_t   = gt['timestamp'].values   - t0
    odom_t = odom['timestamp'].values - t0
    ekf_t  = ekf['timestamp'].values  - t0

    # Interpolate GT to odom and EKF time axes
    gt_x_odom, gt_y_odom = interpolate_to(gt_t, gt['x'].values,
                                           gt['y'].values, odom_t)
    gt_x_ekf,  gt_y_ekf  = interpolate_to(gt_t, gt['x'].values,
                                           gt['y'].values, ekf_t)

    ate_odom = compute_ate(odom['x'].values, odom['y'].values,
                           gt_x_odom, gt_y_odom)
    ate_ekf  = compute_ate(ekf['x'].values,  ekf['y'].values,
                           gt_x_ekf,  gt_y_ekf)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')

    ax.plot(odom_t, ate_odom,
            color='#d62728', linewidth=1.5, linestyle='--',
            alpha=0.8, label=f'Wheel Odometry  (RMSE={np.sqrt(np.mean(ate_odom**2)):.3f} m)')

    ax.plot(ekf_t, ate_ekf,
            color='#1f77b4', linewidth=2.0, linestyle='-',
            label=f'EKF-SLAM  (RMSE={np.sqrt(np.mean(ate_ekf**2)):.3f} m)')

    # Shade the improvement region
    # Interpolate both to a common time axis for shading
    t_common = np.linspace(max(odom_t[0], ekf_t[0]),
                           min(odom_t[-1], ekf_t[-1]), 2000)
    ate_odom_i = np.interp(t_common, odom_t, ate_odom)
    ate_ekf_i  = np.interp(t_common, ekf_t,  ate_ekf)

    ax.fill_between(t_common,
                    np.minimum(ate_odom_i, ate_ekf_i),
                    np.maximum(ate_odom_i, ate_ekf_i),
                    where=ate_ekf_i < ate_odom_i,
                    alpha=0.15, color='#2ca02c',
                    label='EKF improvement region')

    # Mark global correction events as vertical lines
    # Detect sudden drops in EKF ATE (> 0.3m drop in one step)
    ate_diff = np.diff(ate_ekf)
    correction_idx = np.where(ate_diff < -0.3)[0]
    for idx in correction_idx:
        ax.axvline(ekf_t[idx], color='#ff7f0e',
                   linewidth=1.2, linestyle=':', alpha=0.8)

    if len(correction_idx) > 0:
        ax.axvline(ekf_t[correction_idx[0]], color='#ff7f0e',
                   linewidth=1.2, linestyle=':', alpha=0.8,
                   label='Global relocalization event')

    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('ATE [m]', fontsize=12)
    ax.set_title('Absolute Trajectory Error over Time', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=10, loc='upper left',
              framealpha=0.9, edgecolor='#cccccc')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.4, linewidth=0.6)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    path = os.path.join(out_dir, 'ate_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Saved: ate_comparison.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', required=True,
                        help='Directory with ekf_poses.csv, ground_truth.csv, wheel_odom.csv')
    parser.add_argument('--out',     required=True,
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"CSV dir: {args.csv_dir}")
    print(f"Output:  {args.out}")

    # ── Load CSVs ────────────────────────────────────────────
    # Try common naming conventions from extract_bag.py
    def try_load(candidates):
        for name in candidates:
            try:
                return load_csv(args.csv_dir, name)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            f"Could not find any of: {candidates} in {args.csv_dir}")

    print("\nLoading CSVs...")
    gt   = try_load(['ground_truth.csv', 'ground_truth_odom.csv', 'gt.csv'])
    odom = try_load(['wheel_odom.csv', 'cmd_odom.csv', 'odom.csv'])
    ekf  = try_load(['ekf_poses.csv', 'ekf_pose.csv', 'ekf.csv'])

    print(f"  Ground truth: {len(gt)} samples")
    print(f"  Wheel odom:   {len(odom)} samples")
    print(f"  EKF poses:    {len(ekf)} samples")

    # ── Print column names to help debug if needed ───────────
    print(f"\n  GT columns:   {list(gt.columns)}")
    print(f"  Odom columns: {list(odom.columns)}")
    print(f"  EKF columns:  {list(ekf.columns)}")

    # ── Standardise column names ─────────────────────────────
    # Rename common variants to x, y, timestamp
    for df, name in [(gt, 'GT'), (odom, 'Odom'), (ekf, 'EKF')]:
        cols = {c.lower(): c for c in df.columns}
        renames = {}
        for std, variants in [
            ('x',         ['x', 'pos_x', 'position_x', 'pose_x']),
            ('y',         ['y', 'pos_y', 'position_y', 'pose_y']),
            ('timestamp', ['timestamp', 'time', 't', 'secs']),
        ]:
            for v in variants:
                if v in cols and std not in df.columns:
                    renames[cols[v]] = std
                    break
        if renames:
            df.rename(columns=renames, inplace=True)

        missing = [c for c in ['x', 'y', 'timestamp']
                   if c not in df.columns]
        if missing:
            raise ValueError(
                f"{name} CSV missing columns {missing}. "
                f"Available: {list(df.columns)}")

    # ── Generate figures ─────────────────────────────────────
    print("\nGenerating trajectory comparison...")
    plot_trajectory(gt.copy(), odom.copy(), ekf.copy(), args.out)

    print("Generating ATE over time...")
    plot_ate(gt.copy(), odom.copy(), ekf.copy(), args.out)

    print(f"\nDone. Figures saved to: {args.out}")


if __name__ == '__main__':
    main()