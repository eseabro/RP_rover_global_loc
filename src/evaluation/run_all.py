#!/usr/bin/env python3
"""
run_all.py
==========
Master evaluation script. Runs the full evaluation pipeline:
  1. Extract bag to CSVs
  2. Run SLAM metrics
  3. Run relocalization metrics
  4. Run global localization metrics

Usage:
    python3 run_all.py --bag /path/to/bag --out /path/to/results/

If you already extracted CSVs, skip extraction:
    python3 run_all.py --csv_dir /path/to/csvs --out /path/to/results/ --skip_extract
"""

# import argparse
# import os
# import subprocess
# import sys


# def run(cmd, description):
#     print(f"\n{'='*60}")
#     print(f"  {description}")
#     print(f"{'='*60}")
#     result = subprocess.run(cmd, capture_output=False)
#     if result.returncode != 0:
#         print(f"  WARNING: step exited with code {result.returncode}")
#     return result.returncode == 0


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--bag',          default=None, help='Path to ROS2 bag directory')
#     parser.add_argument('--csv_dir',      default=None, help='Directory with pre-extracted CSVs')
#     parser.add_argument('--out',          required=True, help='Output directory for results')
#     parser.add_argument('--skip_extract', action='store_true',
#                         help='Skip bag extraction (use existing CSVs)')
#     parser.add_argument('--rpe_window',   type=float, default=5.0,
#                         help='RPE time window in seconds')
#     parser.add_argument('--error_threshold', type=float, default=2.0,
#                         help='Success threshold for relocalization [m]')
#     args = parser.parse_args()

#     script_dir = os.path.dirname(os.path.abspath(__file__))

#     # Resolve CSV directory
#     if args.csv_dir:
#         csv_dir = args.csv_dir
#     elif args.bag:
#         csv_dir = os.path.join(args.out, 'csv')
#     else:
#         print("ERROR: provide either --bag or --csv_dir")
#         sys.exit(1)

#     os.makedirs(args.out, exist_ok=True)
#     os.makedirs(csv_dir, exist_ok=True)

#     # ── Step 1: Extract ─────────────────────────────────
#     if not args.skip_extract:
#         if not args.bag:
#             print("ERROR: --bag required for extraction")
#             sys.exit(1)
#         run([sys.executable,
#              os.path.join(script_dir, 'extract_bag.py'),
#              '--bag', args.bag,
#              '--out', csv_dir],
#             "Step 1/4: Extracting bag to CSVs")
#     else:
#         print("\nSkipping extraction (--skip_extract)")

#     # Check which CSVs exist
#     ekf_csv    = os.path.join(csv_dir, 'ekf_poses.csv')
#     gt_csv     = os.path.join(csv_dir, 'ground_truth.csv')
#     odom_csv   = os.path.join(csv_dir, 'wheel_odom.csv')
#     global_csv = os.path.join(csv_dir, 'global_poses.csv')
#     stats_csv  = os.path.join(csv_dir, 'matcher_stats.csv')

#     has_ekf    = os.path.exists(ekf_csv)
#     has_gt     = os.path.exists(gt_csv)
#     has_odom   = os.path.exists(odom_csv)
#     has_global = os.path.exists(global_csv)
#     has_stats  = os.path.exists(stats_csv)

#     print(f"\n  Available data:")
#     print(f"    EKF poses:       {'✓' if has_ekf    else '✗'}")
#     print(f"    Ground truth:    {'✓' if has_gt     else '✗ (required for ATE)'}")
#     print(f"    Wheel odometry:  {'✓' if has_odom   else '✗'}")
#     print(f"    Global poses:    {'✓' if has_global else '✗ (required for reloc metrics)'}")
#     print(f"    Matcher stats:   {'✓' if has_stats  else '✗ (optional, for inlier/time metrics)'}")

#     # ── Step 2: SLAM metrics ─────────────────────────────
#     if has_ekf and has_gt:
#         cmd = [sys.executable,
#                os.path.join(script_dir, 'eval_slam_ldm.py'),
#                '--ekf',  ekf_csv,
#                '--gt',   gt_csv,
#                '--out',  os.path.join(args.out, 'slam'),
#                '--rpe_window', str(args.rpe_window)]
#         if has_odom:
#             cmd += ['--odom', odom_csv]
#         run(cmd, "Step 2/4: SLAM Pipeline Metrics (ATE, RPE, 3σ)")
#     else:
#         missing = []
#         if not has_ekf: missing.append('ekf_poses.csv')
#         if not has_gt:  missing.append('ground_truth.csv')
#         print(f"\n  Skipping SLAM metrics — missing: {', '.join(missing)}")

#     # ── Step 3: Relocalization metrics ───────────────────
#     if has_global and has_gt and has_odom:
#         run([sys.executable,
#              os.path.join(script_dir, 'eval_relocalization.py'),
#              '--global_poses',     global_csv,
#              '--gt',               gt_csv,
#              '--odom',             odom_csv,
#              '--out',              os.path.join(args.out, 'relocalization'),
#              '--error_threshold',  str(args.error_threshold)],
#             "Step 3/4: Sequential Relocalization Metrics")
#     else:
#         missing = []
#         if not has_global: missing.append('global_poses.csv')
#         if not has_gt:     missing.append('ground_truth.csv')
#         if not has_odom:   missing.append('wheel_odom.csv')
#         print(f"\n  Skipping relocalization metrics — missing: {', '.join(missing)}")

#     # ── Step 4: Global localization metrics ──────────────
#     if has_global and has_gt:
#         cmd = [sys.executable,
#                os.path.join(script_dir, 'eval_global_loc.py'),
#                '--global_poses', global_csv,
#                '--gt',           gt_csv,
#                '--out',          os.path.join(args.out, 'global_loc')]
#         if has_stats:
#             cmd += ['--stats', stats_csv]
#         run(cmd, "Step 4/4: Global Localization Metrics")
#     else:
#         print(f"\n  Skipping global localization metrics — missing ground truth or global poses")

#     print(f"\n{'='*60}")
#     print(f"  Evaluation complete. Results in: {args.out}")
#     print(f"{'='*60}\n")


# if __name__ == '__main__':
#     main()



import argparse
import os
import subprocess
import sys

def run(cmd, description):
    print(f"\n{'='*60}\n  {description}\n{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', default=None); parser.add_argument('--csv_dir', default=None)
    parser.add_argument('--out', required=True); parser.add_argument('--skip_extract', action='store_true')
    parser.add_argument('--rpe_window', type=float, default=5.0); parser.add_argument('--error_threshold', type=float, default=2.0)
    args = parser.parse_args(); script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = args.csv_dir if args.csv_dir else os.path.join(args.out, 'csv')
    os.makedirs(args.out, exist_ok=True); os.makedirs(csv_dir, exist_ok=True)

    if not args.skip_extract and args.bag:
        run([sys.executable, os.path.join(script_dir, 'extract_bag_ldm.py'), '--bag', args.bag, '--out', csv_dir], "Step 1: Extracting bag (including Landmarks)")

    lms_csv = os.path.join(csv_dir, 'ekf_landmarks.csv')
    has_ekf, has_gt, has_odom, has_lms = os.path.exists(os.path.join(csv_dir, 'ekf_poses.csv')), os.path.exists(os.path.join(csv_dir, 'ground_truth.csv')), os.path.exists(os.path.join(csv_dir, 'wheel_odom.csv')), os.path.exists(lms_csv)

    print(f"\nData: EKF: {'✓' if has_ekf else '✗'}, GT: {'✓' if has_gt else '✗'}, Landmarks: {'✓' if has_lms else '✗'}")

    if has_ekf and has_gt:
        cmd = [sys.executable, os.path.join(script_dir, 'eval_slam_ldm.py'), '--ekf', os.path.join(csv_dir, 'ekf_poses.csv'), '--gt', os.path.join(csv_dir, 'ground_truth.csv'), '--out', os.path.join(args.out, 'slam'), '--rpe_window', str(args.rpe_window)]
        if has_lms: cmd += ['--landmarks', lms_csv]
        if has_odom: cmd += ['--odom', os.path.join(csv_dir, 'wheel_odom.csv')]
        run(cmd, "Step 2: SLAM 3D Metrics & Landmark Plotting")

    print(f"\nEvaluation complete. Results in: {args.out}")

if __name__ == '__main__': main()
