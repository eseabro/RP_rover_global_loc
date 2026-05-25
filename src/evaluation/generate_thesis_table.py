#!/usr/bin/env python3
"""
generate_thesis_table.py
========================
Generates a LaTeX and CSV metrics table for the EKF-SLAM thesis section.
Supports multiple configurations (odom only, EKF only, full pipeline)
and multiple runs per configuration.

Usage — single run:
    python3 generate_thesis_table.py \
        --odom_csv   /home/ws/results/run_ekf_only/csv/wheel_odom.csv \
        --ekf_csv    /home/ws/results/run_ekf_only/csv/ekf_poses.csv \
        --gt_csv     /home/ws/results/run_ekf_only/csv/ground_truth.csv \
        --label      "EKF Only" \
        --out        /home/ws/results/thesis_table/

Usage — multiple configs (run once per config, results accumulate):
    python3 generate_thesis_table.py --odom_csv ... --ekf_csv ... --gt_csv ... --label "Odom Only"    --out ...
    python3 generate_thesis_table.py --odom_csv ... --ekf_csv ... --gt_csv ... --label "EKF Only"     --out ...
    python3 generate_thesis_table.py --odom_csv ... --ekf_csv ... --gt_csv ... \
        --global_poses_csv /home/ws/results/run_full/csv/global_poses.csv \
        --label "Full Pipeline" --out ...

    Then run with --compile_only to generate the final table:
    python3 generate_thesis_table.py --compile_only --out /home/ws/results/thesis_table/
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import json


# ─────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path)
    return df.sort_values('timestamp').reset_index(drop=True)


def align_timestamps(df_query, df_ref, max_dt=0.2):
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


def compute_ate_2d(est, gt):
    """2D XY ATE — unaffected by Z drift."""
    dx = est['x'].values - gt['x'].values
    dy = est['y'].values - gt['y'].values
    errors = np.sqrt(dx**2 + dy**2)
    return {
        'ATE RMSE 2D [m]':   round(float(np.sqrt(np.mean(errors**2))), 4),
        'ATE Mean 2D [m]':   round(float(np.mean(errors)), 4),
        'ATE Max 2D [m]':    round(float(np.max(errors)), 4),
        'ATE Std 2D [m]':    round(float(np.std(errors)), 4),
    }


def compute_rpe(est, gt, delta_t=5.0):
    """Relative Pose Error over fixed time window."""
    times = est['timestamp'].values
    errors = []
    for i in range(len(times)):
        target = times[i] + delta_t
        j = np.searchsorted(times, target)
        if j >= len(times):
            continue
        gt_dx  = gt.iloc[j]['x']  - gt.iloc[i]['x']
        gt_dy  = gt.iloc[j]['y']  - gt.iloc[i]['y']
        est_dx = est.iloc[j]['x'] - est.iloc[i]['x']
        est_dy = est.iloc[j]['y'] - est.iloc[i]['y']
        errors.append(np.sqrt((est_dx - gt_dx)**2 + (est_dy - gt_dy)**2))
    errors = np.array(errors)
    if len(errors) == 0:
        return {'RPE RMSE [m]': float('nan'), 'RPE Mean [m]': float('nan')}
    return {
        'RPE RMSE [m]': round(float(np.sqrt(np.mean(errors**2))), 4),
        'RPE Mean [m]': round(float(np.mean(errors)), 4),
    }


def compute_end_error(est, gt):
    """Distance between final estimated and final GT position (2D)."""
    err = np.hypot(est['x'].iloc[-1] - gt['x'].iloc[-1],
                   est['y'].iloc[-1] - gt['y'].iloc[-1])
    return {'End Position Error [m]': round(float(err), 4)}


def compute_sigma_consistency(est, gt):
    """Percentage of timesteps where XY error is within 3σ covariance."""
    dx = est['x'].values - gt['x'].values
    dy = est['y'].values - gt['y'].values

    results = {}
    for axis, err, cov_col in [('X', dx, 'cov_xx'), ('Y', dy, 'cov_yy')]:
        if cov_col in est.columns:
            sigma3 = 3.0 * np.sqrt(np.abs(est[cov_col].values))
            consistent = float(np.mean(np.abs(err) < sigma3) * 100)
            results[f'3σ Consistent {axis} [%]'] = round(consistent, 1)
        else:
            results[f'3σ Consistent {axis} [%]'] = 'N/A'
    return results


def compute_trajectory_stats(gt):
    """Total distance travelled and duration from GT."""
    dx = np.diff(gt['x'].values)
    dy = np.diff(gt['y'].values)
    dist = float(np.sum(np.sqrt(dx**2 + dy**2)))
    dur  = float(gt['timestamp'].iloc[-1] - gt['timestamp'].iloc[0])
    return {
        'Distance Travelled [m]': round(dist, 2),
        'Duration [s]':           round(dur, 1),
    }


def compute_global_correction_stats(global_poses_csv):
    """Count global relocalization events and mean correction magnitude."""
    if global_poses_csv is None or not os.path.exists(global_poses_csv):
        return {
            'Global Corrections':          'N/A',
            'Mean Correction Magnitude [m]': 'N/A',
        }
    df = pd.read_csv(global_poses_csv)
    n = len(df)
    if n == 0:
        return {'Global Corrections': 0, 'Mean Correction Magnitude [m]': 'N/A'}

    if 'x' in df.columns and 'y' in df.columns and len(df) > 1:
        dx = np.diff(df['x'].values)
        dy = np.diff(df['y'].values)
        mag = round(float(np.mean(np.sqrt(dx**2 + dy**2))), 3)
    else:
        mag = 'N/A'

    return {
        'Global Corrections':            n,
        'Mean Correction Magnitude [m]': mag,
    }


# ─────────────────────────────────────────────────────────────
# Compute all metrics for one configuration
# ─────────────────────────────────────────────────────────────

def compute_all_metrics(odom_csv, ekf_csv, gt_csv,
                        global_poses_csv=None, rpe_window=5.0):
    gt   = load_csv(gt_csv)
    odom = load_csv(odom_csv)
    ekf  = load_csv(ekf_csv)

    # Align timestamps
    ekf_al,  gt_al_ekf  = align_timestamps(ekf,  gt)
    odom_al, gt_al_odom = align_timestamps(odom, gt)

    metrics = {}

    # Trajectory stats (from GT)
    metrics.update(compute_trajectory_stats(gt_al_ekf))

    # Odometry metrics
    odom_ate  = compute_ate_2d(odom_al, gt_al_odom)
    odom_end  = compute_end_error(odom_al, gt_al_odom)
    metrics['Odom ATE RMSE 2D [m]']     = odom_ate['ATE RMSE 2D [m]']
    metrics['Odom End Position Error [m]'] = odom_end['End Position Error [m]']

    # EKF metrics
    metrics.update(compute_ate_2d(ekf_al, gt_al_ekf))
    metrics.update(compute_rpe(ekf_al, gt_al_ekf, delta_t=rpe_window))
    metrics.update(compute_end_error(ekf_al, gt_al_ekf))
    metrics.update(compute_sigma_consistency(ekf_al, gt_al_ekf))

    # ATE improvement
    odom_rmse = metrics['Odom ATE RMSE 2D [m]']
    ekf_rmse  = metrics['ATE RMSE 2D [m]']
    if isinstance(odom_rmse, float) and odom_rmse > 0:
        improvement = round((odom_rmse - ekf_rmse) / odom_rmse * 100, 1)
        metrics['ATE Improvement vs Odom [%]'] = improvement
    else:
        metrics['ATE Improvement vs Odom [%]'] = 'N/A'

    # Global correction stats
    metrics.update(compute_global_correction_stats(global_poses_csv))

    return metrics


# ─────────────────────────────────────────────────────────────
# LaTeX table generator
# ─────────────────────────────────────────────────────────────

def generate_latex_table(all_results: dict, out_dir: str):
    """
    Generates a LaTeX longtable with one column per configuration.
    all_results: {label: {metric: value}}
    """
    labels  = list(all_results.keys())
    # Collect all metric keys in order
    all_keys = []
    seen = set()
    for metrics in all_results.values():
        for k in metrics:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Group metrics for visual separation
    groups = {
        'Trajectory': ['Distance Travelled [m]', 'Duration [s]'],
        'EKF Performance': [
            'ATE RMSE 2D [m]', 'ATE Mean 2D [m]', 'ATE Max 2D [m]', 'ATE Std 2D [m]',
            'RPE RMSE [m]', 'RPE Mean [m]',
            'End Position Error [m]',
            'ATE Improvement vs Odom [%]',
        ],
        'Baseline': ['Odom ATE RMSE 2D [m]', 'Odom End Position Error [m]'],
        'Filter Consistency': ['3σ Consistent X [%]', '3σ Consistent Y [%]'],
        'Global Relocalization': ['Global Corrections', 'Mean Correction Magnitude [m]'],
    }

    n_cols = len(labels)
    col_fmt = 'l' + 'c' * n_cols

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{EKF-SLAM Evaluation Metrics}')
    lines.append(r'\label{tab:ekf_results}')
    lines.append(r'\begin{tabular}{' + col_fmt + r'}')
    lines.append(r'\toprule')

    # Header
    header = 'Metric & ' + ' & '.join(
        r'\textbf{' + l.replace('_', r'\_') + '}' for l in labels
    ) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')

    # Rows by group
    for group_name, group_keys in groups.items():
        present = [k for k in group_keys if k in seen]
        if not present:
            continue
        lines.append(r'\multicolumn{' + str(n_cols + 1) + r'}{l}{\textit{' +
                     group_name + r'}} \\')
        for k in present:
            row_vals = []
            for label in labels:
                v = all_results[label].get(k, '—')
                if isinstance(v, float):
                    row_vals.append(f'{v:.4f}')
                else:
                    row_vals.append(str(v))
            row = k.replace('[m]', '[m]').replace('%', r'\%') + \
                  ' & ' + ' & '.join(row_vals) + r' \\'
            lines.append(row)
        lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    latex = '\n'.join(lines)
    path = os.path.join(out_dir, 'ekf_metrics_table.tex')
    with open(path, 'w') as f:
        f.write(latex)
    print(f"  Saved: ekf_metrics_table.tex")
    return latex


def generate_csv_table(all_results: dict, out_dir: str):
    """Simple CSV version of the table."""
    labels = list(all_results.keys())
    all_keys = []
    seen = set()
    for metrics in all_results.values():
        for k in metrics:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    rows = []
    for k in all_keys:
        row = {'Metric': k}
        for label in labels:
            row[label] = all_results[label].get(k, '—')
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'ekf_metrics_table.csv')
    df.to_csv(path, index=False)
    print(f"  Saved: ekf_metrics_table.csv")

    # Pretty print to terminal
    print('\n' + '=' * (30 + 15 * len(labels)))
    print(f"  {'Metric':<35}" + ''.join(f"{l:>15}" for l in labels))
    print('=' * (30 + 15 * len(labels)))
    for _, row in df.iterrows():
        metric = str(row['Metric'])[:34]
        vals   = ''.join(f"{str(row[l]):>15}" for l in labels)
        print(f"  {metric:<35}{vals}")
    print('=' * (30 + 15 * len(labels)))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ekf_csv',          default=None)
    parser.add_argument('--odom_csv',         default=None)
    parser.add_argument('--gt_csv',           default=None)
    parser.add_argument('--global_poses_csv', default=None,
                        help='global_poses.csv from extract_bag.py (optional)')
    parser.add_argument('--label',            default='Run',
                        help='Name for this configuration in the table')
    parser.add_argument('--rpe_window',       type=float, default=5.0)
    parser.add_argument('--out',              required=True)
    parser.add_argument('--compile_only',     action='store_true',
                        help='Skip computation, just regenerate table from saved results')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, 'all_results.json')

    # Load existing results if any
    all_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")

    if not args.compile_only:
        if not all([args.ekf_csv, args.odom_csv, args.gt_csv]):
            print("ERROR: --ekf_csv, --odom_csv, --gt_csv are required unless --compile_only")
            return

        print(f"\nComputing metrics for: {args.label}")
        metrics = compute_all_metrics(
            odom_csv         = args.odom_csv,
            ekf_csv          = args.ekf_csv,
            gt_csv           = args.gt_csv,
            global_poses_csv = args.global_poses_csv,
            rpe_window       = args.rpe_window,
        )
        all_results[args.label] = metrics

        # Save updated results
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to all_results.json")

    if len(all_results) == 0:
        print("No results to compile.")
        return

    # Generate table
    print(f"\nGenerating table from {len(all_results)} configuration(s)...")
    generate_csv_table(all_results, args.out)
    generate_latex_table(all_results, args.out)

    print(f"\nDone. Outputs in: {args.out}")


if __name__ == '__main__':
    main()