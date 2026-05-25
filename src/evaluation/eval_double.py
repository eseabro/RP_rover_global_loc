import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# --- Utilities from eval_slam_ldm.py ---
def align_timestamps(df_query, df_ref, max_dt=0.2):
    ref_times = df_ref['timestamp'].values
    tree = KDTree(ref_times.reshape(-1, 1))
    aligned_query, aligned_ref = [], []
    for _, row in df_query.iterrows():
        dist, idx = tree.query([[row['timestamp']]])
        if dist[0] < max_dt:
            aligned_query.append(row)
            aligned_ref.append(df_ref.iloc[idx[0]])
    return pd.DataFrame(aligned_query).reset_index(drop=True), \
           pd.DataFrame(aligned_ref).reset_index(drop=True)

# 1. Load Data
gt = pd.read_csv('run_with_reloc/csv/ground_truth.csv')
ekf_with = pd.read_csv('run_with_reloc/csv/ekf_poses.csv')
ekf_without = pd.read_csv('run_without_reloc/csv/ekf_poses.csv')

# 2. Align all runs to Ground Truth for fair comparison
ekf_w_al, gt_w_al = align_timestamps(ekf_with, gt)
ekf_no_al, gt_no_al = align_timestamps(ekf_without, gt)

# --- PLOT 1: TRAJECTORY OVERLAY ---
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(gt['x'], gt['y'], 'g-', linewidth=2, label='Ground Truth', zorder=1)
ax1.plot(ekf_without['x'], ekf_without['y'], 'r--', label='EKF (No Reloc)', alpha=0.6)
ax1.plot(ekf_with['x'], ekf_with['y'], 'b-', label='EKF (With Reloc)', alpha=0.9)
ax1.set_title('Trajectory Comparison: With vs Without Relocalization')
ax1.set_aspect('equal'); ax1.legend(); ax1.grid(True, alpha=0.3)
plt.savefig('combined_trajectories.png', dpi=200)

# --- PLOT 2: DUAL SIGMA CONSISTENCY ---
fig2, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
t0 = gt_w_al['timestamp'].iloc[0]

# Metrics to plot: Distance Error and Yaw Error
for i, mode in enumerate(['dist', 'yaw']):
    ax = axes[i]
    
    if mode == 'dist':
        # Calculate Distance Errors
        err_with = np.sqrt((ekf_w_al['x'] - gt_w_al['x'])**2 + (ekf_w_al['y'] - gt_w_al['y'])**2)
        err_no = np.sqrt((ekf_no_al['x'] - gt_no_al['x'])**2 + (ekf_no_al['y'] - gt_no_al['y'])**2)
        # 3rd-party sigma bounds (usually taken from the active EKF run)
        sigma = 3 * np.sqrt(np.abs(ekf_w_al['cov_xx']) + np.abs(ekf_w_al['cov_yy']))
        ax.fill_between(ekf_w_al['timestamp'] - t0, 0, sigma, color='orange', alpha=0.2, label='3σ Uncertainty Bound')
        ax.set_ylabel('XY Pos Error [m]')
    else:
        # Calculate Yaw Errors (with wrap-around normalization)
        raw_err_w = ekf_w_al['yaw'] - gt_w_al['yaw']
        err_with = np.arctan2(np.sin(raw_err_w), np.cos(raw_err_w))
        raw_err_no = ekf_no_al['yaw'] - gt_no_al['yaw']
        err_no = np.arctan2(np.sin(raw_err_no), np.cos(raw_err_no))
        sigma = 3 * np.sqrt(np.abs(ekf_w_al['cov_yaw']))
        ax.fill_between(ekf_w_al['timestamp'] - t0, -sigma, sigma, color='orange', alpha=0.2, label='3σ Uncertainty Bound')
        ax.set_ylabel('Yaw Error [rad]')

    # Plot both error lines
    ax.plot(ekf_no_al['timestamp'] - t0, err_no, 'r--', alpha=0.6, label='Error (No Reloc)')
    ax.plot(ekf_w_al['timestamp'] - t0, err_with, 'b-', alpha=0.9, label='Error (With Reloc)')
    
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

axes[1].set_xlabel('Time [s]')
fig2.suptitle('Scientific Comparison: Error Consistency vs 3σ Bounds', fontsize=14)
plt.tight_layout()
plt.savefig('combined_sigma_consistency.png', dpi=200)

print("Saved combined_trajectories.png and combined_sigma_consistency.png!")