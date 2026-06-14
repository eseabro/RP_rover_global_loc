# #!/usr/bin/env python3
# """
# eval_slam.py
# ============
# Evaluates the full SLAM pipeline against ground truth, now including Landmarks.
# """

# import argparse
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.spatial import KDTree

# # ─────────────────────────────────────────────
# # Utilities
# # ─────────────────────────────────────────────

# def load_csv(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"CSV not found: {path}")
#     df = pd.read_csv(path)
#     df = df.sort_values('timestamp').reset_index(drop=True)
#     return df

# # ... [Keep align_timestamps, compute_ate, compute_rpe, compute_sigma_consistency exactly as they were] ...

# def align_timestamps(df_query, df_ref, max_dt=0.2):
#     ref_times = df_ref['timestamp'].values
#     tree = KDTree(ref_times.reshape(-1, 1))
#     aligned_query = []
#     aligned_ref = []
#     for _, row in df_query.iterrows():
#         dist, idx = tree.query([[row['timestamp']]])
#         if dist[0] < max_dt:
#             aligned_query.append(row)
#             aligned_ref.append(df_ref.iloc[idx[0]])
#     return pd.DataFrame(aligned_query).reset_index(drop=True), \
#            pd.DataFrame(aligned_ref).reset_index(drop=True)

# def compute_ate(est, gt):
#     dx = est['x'].values - gt['x'].values
#     dy = est['y'].values - gt['y'].values
#     dz = est['z'].values - gt['z'].values  # Added Z
    
#     errors_2d = np.sqrt(dx**2 + dy**2)
#     errors_3d = np.sqrt(dx**2 + dy**2 + dz**2)
#     errors_z  = np.abs(dz) # Absolute Z error
    
#     return errors_3d, {
#         'ATE_RMSE_3D': float(np.sqrt(np.mean(errors_3d**2))),
#         'ATE_RMSE_Z':  float(np.sqrt(np.mean(errors_z**2))), # Specific Z RMSE
#         'ATE_mean_Z':  float(np.mean(errors_z)),
#         'ATE_max_Z':   float(np.max(errors_z)),
#         'ATE_final_Z': float(errors_z[-1]) if len(errors_z) > 0 else float('nan'),
#         'ATE_mean_3D': float(np.mean(errors_3d)),
#     }

# def compute_rpe(est, gt, delta_t=5.0):
#     times = est['timestamp'].values
#     errors = []
#     for i in range(len(times)):
#         target = times[i] + delta_t
#         j = np.searchsorted(times, target)
#         if j >= len(times): continue
#         gt_dx = gt.iloc[j]['x'] - gt.iloc[i]['x']
#         gt_dy = gt.iloc[j]['y'] - gt.iloc[i]['y']
#         est_dx = est.iloc[j]['x'] - est.iloc[i]['x']
#         est_dy = est.iloc[j]['y'] - est.iloc[i]['y']
#         err = np.sqrt((est_dx - gt_dx)**2 + (est_dy - gt_dy)**2)
#         errors.append(err)
#     errors = np.array(errors)
#     return errors, {
#         'RPE_RMSE': float(np.sqrt(np.mean(errors**2))) if len(errors) > 0 else float('nan'),
#         'RPE_mean': float(np.mean(errors)) if len(errors) > 0 else float('nan'),
#         'RPE_max':  float(np.max(errors)) if len(errors) > 0 else float('nan'),
#     }

# def compute_sigma_consistency(est, gt):
#     dx = est['x'].values - gt['x'].values
#     dy = est['y'].values - gt['y'].values
#     dz = est['z'].values - gt['z'].values
    
#     # 3-sigma bounds for X, Y, and Z
#     sigma_x = 3.0 * np.sqrt(np.abs(est['cov_xx'].values))
#     sigma_y = 3.0 * np.sqrt(np.abs(est['cov_yy'].values))
#     sigma_z = 3.0 * np.sqrt(np.abs(est['cov_zz'].values)) # Added Z

#     consistent_x = np.abs(dx) < sigma_x
#     consistent_y = np.abs(dy) < sigma_y
#     consistent_z = np.abs(dz) < sigma_z
#     consistent_all = consistent_x & consistent_y & consistent_z

#     return {
#         'sigma_x': sigma_x, 'sigma_y': sigma_y, 'sigma_z': sigma_z,
#         'error_x': dx, 'error_y': dy, 'error_z': dz,
#         'consistent_z_pct': float(np.mean(consistent_z) * 100),
#         'consistent_all_pct': float(np.mean(consistent_all) * 100),
#     }

# # ─────────────────────────────────────────────
# # Updated Plotting
# # ─────────────────────────────────────────────

# def plot_trajectories(ekf, gt, odom, landmarks, out_dir):
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     # 1. Plot Landmarks first (Background)
#     if landmarks is not None and not landmarks.empty:
#         # We only want the unique final position of each landmark ID
#         # (Otherwise they will look like lines if the EKF updated them)
#         final_landmarks = landmarks.sort_values('timestamp').groupby('id').tail(1)
        
#         # Split into Active and Archived based on alpha from your EKF code
#         active = final_landmarks[final_landmarks['alpha'] > 0.5]
#         archived = final_landmarks[final_landmarks['alpha'] <= 0.5]
        
#         ax.scatter(archived['x'], archived['y'], c='gray', s=20, alpha=0.3, 
#                    marker='o', label='Archived Landmarks', edgecolors='none')
#         ax.scatter(active['x'], active['y'], c='orange', s=40, alpha=0.8, 
#                    marker='*', label='Active Landmarks', edgecolors='black', linewidths=0.5)

#     # 2. Plot Trajectories
#     ax.plot(gt['x'], gt['y'], 'g-', linewidth=2, label='Ground Truth', zorder=5)
#     if odom is not None:
#         ax.plot(odom['x'], odom['y'], 'r--', linewidth=1.5, label='Wheel Odometry', alpha=0.6, zorder=4)
#     ax.plot(ekf['x'], ekf['y'], 'b-', linewidth=1.5, label='EKF Estimate', alpha=0.9, zorder=6)

#     # Mark start and end
#     ax.plot(gt['x'].iloc[0], gt['y'].iloc[0], 'ko', markersize=8, label='Start', zorder=10)
#     ax.plot(gt['x'].iloc[-1], gt['y'].iloc[-1], 'g^', markersize=8, label='End (GT)', zorder=10)
#     ax.plot(ekf['x'].iloc[-1], ekf['y'].iloc[-1], 'b^', markersize=8, label='End (EKF)', zorder=10)

#     ax.set_xlabel('X [m]')
#     ax.set_ylabel('Y [m]')
#     ax.set_title('SLAM Trajectory & Landmark Map')
#     ax.legend(loc='upper right', fontsize='small', framealpha=0.8)
#     ax.grid(True, alpha=0.3)
#     ax.set_aspect('equal')
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, 'trajectories.png'), dpi=150)
#     plt.close()
#     print("  Saved: trajectories.png (with landmarks)")

# # ... [Keep plot_ate_over_time, plot_sigma_consistency, plot_covariance_over_time, save_summary_table as they were] ...

# def plot_ate_over_time(times, ate_errors_ekf, ate_errors_odom, out_dir, times_odom=None):
#     fig, ax = plt.subplots(figsize=(12, 5))
#     t0 = times[0]
#     t_rel = times - t0
#     ax.plot(t_rel, ate_errors_ekf, 'b-', linewidth=1.5, label='EKF ATE')
#     if ate_errors_odom is not None:
#         t_odom = times_odom if times_odom is not None else times
#         t_odom_rel = t_odom - t_odom[0]
#         ax.plot(t_odom_rel, ate_errors_odom, 'r--', linewidth=1.5, label='Odometry ATE', alpha=0.8)
#     ax.set_xlabel('Time [s]')
#     ax.set_ylabel('Position Error [m]')
#     ax.set_title('Absolute Trajectory Error over Time')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, 'ate_over_time.png'), dpi=150)
#     plt.close()

# def plot_sigma_consistency(times, sigma_data, out_dir):
#     fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True) # Change to 3 subplots
#     t0 = times[0]
#     t_rel = times - t0
    
#     for i, (axis_name, err_key, sig_key) in enumerate([
#         ('X', 'error_x', 'sigma_x'), 
#         ('Y', 'error_y', 'sigma_y'),
#         ('Z', 'error_z', 'sigma_z')]):
        
#         ax = axes[i]
#         err, sig = sigma_data[err_key], sigma_data[sig_key]
#         ax.plot(t_rel, err, 'b-', linewidth=1.0, label=f'Error {axis_name}')
#         ax.fill_between(t_rel, -sig, sig, alpha=0.2, color='orange', label='3σ bound')
        
#         violations = np.abs(err) > sig
#         if np.any(violations):
#             ax.fill_between(t_rel, err, 0, where=violations, color='red', alpha=0.4)
        
#         ax.set_ylabel(f'{axis_name} Error [m]')
#         ax.legend(loc='upper right')
#         ax.grid(True, alpha=0.3)
    
#     axes[2].set_xlabel('Time [s]')
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, 'sigma_consistency.png'), dpi=150)
#     plt.close()

# def plot_covariance_over_time(ekf, times, out_dir):
#     fig, ax = plt.subplots(figsize=(12, 5))
#     t0 = times[0]
#     t_rel = times - t0
#     ax.plot(t_rel, np.sqrt(ekf['cov_xx'].values), 'b-', label='σ_x')
#     ax.plot(t_rel, np.sqrt(ekf['cov_yy'].values), 'r-', label='σ_y')
#     ax.plot(t_rel, np.sqrt(ekf['cov_yaw'].values), 'g-', label='σ_yaw')
#     ax.set_xlabel('Time [s]')
#     ax.set_ylabel('1σ Uncertainty [m or rad]')
#     ax.legend(); ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, 'covariance_over_time.png'), dpi=150)
#     plt.close()

# def save_summary_table(stats, out_dir):
#     rows = [{'Metric': k, 'Value': f'{v:.4f}' if isinstance(v, float) else v} for k, v in stats.items()]
#     df = pd.DataFrame(rows)
#     df.to_csv(os.path.join(out_dir, 'slam_summary.csv'), index=False)

# # ─────────────────────────────────────────────
# # Updated Main
# # ─────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ekf',  required=True, help='ekf_poses.csv')
#     parser.add_argument('--gt',   required=True, help='ground_truth.csv')
#     parser.add_argument('--landmarks', default=None, help='ekf_landmarks.csv (optional)')
#     parser.add_argument('--odom', default=None,  help='wheel_odom.csv (optional)')
#     parser.add_argument('--out',  required=True, help='Output directory')
#     parser.add_argument('--rpe_window', type=float, default=5.0)
#     args = parser.parse_args()

#     os.makedirs(args.out, exist_ok=True)

#     print("Loading data...")
#     ekf = load_csv(args.ekf)
#     gt  = load_csv(args.gt)
#     odom = load_csv(args.odom) if args.odom else None
#     lms  = load_csv(args.landmarks) if args.landmarks else None

#     # Align timestamps
#     print("\nAligning timestamps...")
#     ekf_aligned, gt_aligned = align_timestamps(ekf, gt)

#     odom_aligned = None
#     gt_aligned_odom = None
#     if odom is not None:
#         odom_aligned, gt_aligned_odom = align_timestamps(odom, gt)

#     times = ekf_aligned['timestamp'].values

#     # Metric Computations
#     ate_errors_ekf, ate_stats = compute_ate(ekf_aligned, gt_aligned)
#     ate_errors_odom, ate_stats_odom = (None, {})
#     if odom_aligned is not None:
#         ate_errors_odom, ate_stats_odom = compute_ate(odom_aligned, gt_aligned_odom)

#     _, rpe_stats = compute_rpe(ekf_aligned, gt_aligned, delta_t=args.rpe_window)
#     rpe_stats_odom = compute_rpe(odom_aligned, gt_aligned_odom, delta_t=args.rpe_window)[1] if odom_aligned is not None else {}
#     sigma_data = compute_sigma_consistency(ekf_aligned, gt_aligned)

#     # Compile summary
#     summary = {
#         'EKF ATE RMSE 3D [m]': ate_stats['ATE_RMSE_3D'],
#         'EKF ATE RMSE Z [m]':  ate_stats['ATE_RMSE_Z'], # New
#         'EKF RPE RMSE [m]':    rpe_stats['RPE_RMSE'],
#         '3σ Consistent Z [%]': sigma_data['consistent_z_pct'], # New
#         '3σ Consistent All [%]': sigma_data['consistent_all_pct'],
#         'Landmarks Discovered': len(lms['id'].unique()) if lms is not None else 0,
#         'Duration [s]': float(times[-1] - times[0]),
#     }

#     # Plots
#     print("\nGenerating plots...")
#     plot_trajectories(ekf_aligned, gt_aligned, odom_aligned, lms, args.out)
    
#     times_odom = odom_aligned['timestamp'].values if odom_aligned is not None else None
#     plot_ate_over_time(times, ate_errors_ekf, ate_errors_odom, args.out, times_odom=times_odom)
#     plot_sigma_consistency(times, sigma_data, args.out)
#     plot_covariance_over_time(ekf_aligned, times, args.out)
#     save_summary_table(summary, args.out)

# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
"""
eval_slam_ldm.py
================
Full 3D SLAM evaluation: Landmarks, Z-axis, and EKF vs Odom ATE comparison.
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
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

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

def compute_ate(est, gt):
    """Computes 2D (XY) and 3D Absolute Trajectory Error.
    Returns 2D errors for plotting to avoid Z drift inflation."""
    dx = est['x'].values - gt['x'].values
    dy = est['y'].values - gt['y'].values
    dz = est['z'].values - gt['z'].values
    errors_2d = np.sqrt(dx**2 + dy**2)
    errors_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    return errors_2d, {
        'RMSE_2D': float(np.sqrt(np.mean(errors_2d**2))),
        'Mean_2D': float(np.mean(errors_2d)),
        'Max_2D':  float(np.max(errors_2d)),
        'RMSE_3D': float(np.sqrt(np.mean(errors_3d**2))),
        'Mean_3D': float(np.mean(errors_3d)),
        'Max_3D':  float(np.max(errors_3d)),
    }

# ─── UPDATED PLOT FUNCTIONS ──────────────────────────────────────────────────

def plot_ate_over_time(ekf_times, ekf_errors, odom_times, odom_errors, out_dir, df_no_reloc=None):
    """Plots both EKF and Odometry ATE for comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # EKF Line (Blue)
    ax.plot(ekf_times - ekf_times[0], ekf_errors, 'b-', label='EKF ATE (2D XY)', linewidth=1.5)
    
    # Odometry Line (Red dashed)
    if odom_errors is not None:
        ax.plot(odom_times - odom_times[0], odom_errors, 'r--', label='Wheel Odom ATE (2D XY)', alpha=0.7)
        
    

        # Inside your plotting block (adjust column names 'x' and 'y' to match your CSV format)
    if df_no_reloc is not None:
        ax.plot(df_no_reloc['x'], df_no_reloc['y'], label='EKF (No Reloc)', color='purple', linestyle='--')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position Error [m]')
    ax.set_title('Absolute Trajectory Error (ATE) Comparison — XY Plane')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ate_over_time.png'), dpi=150)
    plt.close()

def plot_trajectories(ekf, gt, odom, landmarks, out_dir, df_no_reloc=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ══════════════════════════════════════════════════════════════
    # UNIFIED LANDMARK PLOTTING
    # ══════════════════════════════════════════════════════════════
    # if landmarks is not None and not landmarks.empty:
    #     # Still group by ID to get the final estimated position of each rock
    #     lms = landmarks.sort_values('timestamp').groupby('id').tail(1)
        
    #     # Plot all rocks using the same style (Orange stars)
    #     ax.scatter(lms['x'], lms['y'], 
    #                c='orange', 
    #                s=30, 
    #                marker='*', 
    #                alpha=0.8, 
    #                edgecolors='black', 
    #                linewidths=0.5,
    #                label='Estimated Landmarks',
    #                zorder=3)
    # ══════════════════════════════════════════════════════════════
    if df_no_reloc is not None:
        ax.plot(df_no_reloc['x'], df_no_reloc['y'], label='EKF (No Loc)', color='purple', zorder=2)
    
    ax.plot(gt['x'], gt['y'], 'g-', linewidth=2, label='Ground Truth', zorder=5)
    if odom is not None: 
        ax.plot(odom['x'], odom['y'], 'r--', alpha=0.5, label='Wheel Odom', zorder=4)
    
    ax.plot(ekf['x'], ekf['y'], 'b-', alpha=0.8, label='EKF Estimate with Loc', zorder=6)
    
    ax.set_title('Top-Down Trajectory & Landmarks')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.savefig(os.path.join(out_dir, 'trajectories_2d.png'), dpi=150)
    plt.close()

def plot_z_trajectory(ekf, gt, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    t0 = gt['timestamp'].iloc[0]
    ax.plot(gt['timestamp']-t0, gt['z'], 'g-', label='GT Altitude', linewidth=2)
    ax.plot(ekf['timestamp']-t0, ekf['z'], 'b--', label='EKF Altitude', linewidth=1.5)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Z [m]'); ax.set_title('Vertical Profile (Z-axis)'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'z_trajectory.png'), dpi=150); plt.close()

def plot_sigma_consistency(times, ekf, gt, out_dir, relocs=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t0 = times[0]
    t_rel = times - t0
    
    # 1. Distance Error
    dx = ekf['x'] - gt['x']
    dy = ekf['y'] - gt['y']
    dist_error = np.sqrt(dx**2 + dy**2)
    sigma_dist = 3 * np.sqrt(abs(ekf['cov_xx']) + abs(ekf['cov_yy']))
    
    # 2. Yaw Error
    raw_dyaw = ekf['yaw'] - gt['yaw']
    dyaw = np.arctan2(np.sin(raw_dyaw), np.cos(raw_dyaw))
    sigma_yaw = 3 * np.sqrt(abs(ekf['cov_yaw']))
    
    errors = [dist_error, dyaw]
    sigmas = [sigma_dist, sigma_yaw]
    axis_names = ['Distance Error', 'Yaw Error']
    axis_units = ['[m]', '[rad]']
    
    for i, ax in enumerate(axes):
        # Plot the error and bounds first so Matplotlib calculates the Y-axis scale
        ax.plot(t_rel, errors[i], 'b-', label='Error')
        if i == 0:
            ax.fill_between(t_rel, 0, sigmas[i], alpha=0.2, color='orange', label='3σ Bound')
        else:
            ax.fill_between(t_rel, -sigmas[i], sigmas[i], alpha=0.2, color='orange', label='3σ Bound')
            
        # Get the dynamically calculated top of the Y-axis
        y_bottom, y_top = ax.get_ylim()
            
        # ══════════════════════════════════════════════════════════════
        # THE FIX: Drop lines from the top down to the 3-sigma envelope
        # ══════════════════════════════════════════════════════════════
        if relocs is not None and not relocs.empty:
            added_label = False
            for t_reloc in relocs['timestamp'].values:
                reloc_t_rel = t_reloc - t0
                if 0 <= reloc_t_rel <= t_rel[-1]:
                    # Interpolate exactly how high the orange bound is at this millisecond
                    current_sigma = np.interp(reloc_t_rel, t_rel, sigmas[i])
                    
                    # Draw a vertical line dropping from y_top down to current_sigma
                    ax.vlines(x=reloc_t_rel, ymin=current_sigma, ymax=y_top, 
                              colors='purple', linestyles='--', alpha=0.6, linewidth=1.5,
                              label='Global Reloc' if not added_label else "")
                    added_label = True

        # Lock the Y-axis top so drawing our lines doesn't accidentally stretch the graph upward
        ax.set_ylim(top=y_top)

        ax.set_ylabel(f'{axis_names[i]} {axis_units[i]}') 
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sigma_consistency.png'), dpi=150)
    plt.close(fig)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ekf', required=True); parser.add_argument('--gt', required=True)
    parser.add_argument('--landmarks', default=None); parser.add_argument('--odom', default=None)
    parser.add_argument('--out', required=True); parser.add_argument('--rpe_window', type=float, default=5.0)
    parser.add_argument('--global_poses', required=True, help='global_poses.csv')
    parser.add_argument('--ekf_no_reloc', default=None, help='Path to the original EKF CSV (without reloc) from another directory')
    args = parser.parse_args(); os.makedirs(args.out, exist_ok=True)
    if args.ekf_no_reloc and os.path.exists(args.ekf_no_reloc):
        # Assuming pandas is used to load your CSVs
        df_no_reloc = pd.read_csv(args.ekf_no_reloc)
    
    ekf, gt = load_csv(args.ekf), load_csv(args.gt)
    odom = load_csv(args.odom) if args.odom else None
    lms = load_csv(args.landmarks) if args.landmarks else None
    # relocs = load_csv(args.global_poses) if args.global_poses else None
    relocs = None
    # Align and compute EKF Errors
    ekf_al, gt_al_ekf = align_timestamps(ekf, gt)
    ate_err_ekf, stats_ekf = compute_ate(ekf_al, gt_al_ekf)
    
    # Align and compute Odom Errors
    ate_err_odom, odom_times_aligned = None, None
    odom_al = None
    if odom is not None:
        odom_al, gt_al_odom = align_timestamps(odom, gt)
        ate_err_odom, _ = compute_ate(odom_al, gt_al_odom)
        odom_times_aligned = odom_al['timestamp'].values

    # Plotting
    plot_trajectories(ekf_al, gt_al_ekf, odom_al, lms, args.out, df_no_reloc=df_no_reloc if args.ekf_no_reloc else None)
    plot_z_trajectory(ekf_al, gt_al_ekf, args.out)
    plot_ate_over_time(ekf_al['timestamp'].values, ate_err_ekf, odom_times_aligned, ate_err_odom, args.out)
    plot_sigma_consistency(ekf_al['timestamp'].values, ekf_al, gt_al_ekf, args.out, relocs)
    
    summary = {
        'EKF_ATE_RMSE_2D': stats_ekf['RMSE_2D'],
        'EKF_ATE_RMSE_3D': stats_ekf['RMSE_3D'],
        'Rocks_Found': len(lms['id'].unique()) if lms is not None else 0
    }
    pd.DataFrame([{'Metric': k, 'Value': v} for k, v in summary.items()]).to_csv(os.path.join(args.out, 'slam_summary.csv'), index=False)

if __name__ == '__main__': main()