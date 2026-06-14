#!/usr/bin/env python3
"""
run_all.py
==========
Master evaluation script. Runs the full evaluation pipeline:
  1. Extract bag to CSVs (including landmarks)
  2. Run SLAM metrics (Trajectory plotting with landmarks, ATE, RPE, 3-sigma)
  3. Run relocalization metrics
  4. Run global localization metrics

Usage:
    python3 run_all.py --bag /path/to/bag --out /path/to/results/
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
#         # Using your specific extraction script that handles landmarks
#         run([sys.executable,
#              os.path.join(script_dir, 'extract_bag_ldm.py'),
#              '--bag', args.bag,
#              '--out', csv_dir],
#             "Step 1/4: Extracting bag to CSVs (including landmarks)")
#     else:
#         print("\nSkipping extraction (--skip_extract)")

#     # ── Define CSV Paths ────────────────────────────────
#     ekf_csv       = os.path.join(csv_dir, 'ekf_poses.csv')
#     gt_csv        = os.path.join(csv_dir, 'ground_truth.csv')
#     odom_csv      = os.path.join(csv_dir, 'wheel_odom.csv')
#     global_csv    = os.path.join(csv_dir, 'global_poses.csv')
#     stats_csv     = os.path.join(csv_dir, 'matcher_stats.csv')
#     landmarks_csv = os.path.join(csv_dir, 'ekf_landmarks.csv') # New

#     has_ekf       = os.path.exists(ekf_csv)
#     has_gt        = os.path.exists(gt_csv)
#     has_odom      = os.path.exists(odom_csv)
#     has_global    = os.path.exists(global_csv)
#     has_stats     = os.path.exists(stats_csv)
#     has_landmarks = os.path.exists(landmarks_csv) # New

#     print(f"\n  Available data status:")
#     print(f"    EKF poses:       {'✓' if has_ekf       else '✗'}")
#     print(f"    Landmarks:       {'✓' if has_landmarks else '✗'}")
#     print(f"    Ground truth:    {'✓' if has_gt        else '✗ (required for ATE)'}")
#     print(f"    Wheel odometry:  {'✓' if has_odom      else '✗'}")
#     print(f"    Global poses:    {'✓' if has_global    else '✗ (required for reloc metrics)'}")
#     print(f"    Matcher stats:   {'✓' if has_stats     else '✗'}")

#     # ── Step 2: SLAM metrics ─────────────────────────────
#     if has_ekf and has_gt:
#         cmd = [sys.executable,
#                os.path.join(script_dir, 'eval_slam_ldm.py'),
#                '--ekf',  ekf_csv,
#                '--gt',   gt_csv,
#                '--out',  os.path.join(args.out, 'slam'),
#                '--rpe_window', str(args.rpe_window)]
        
#         # Pass landmarks if they were successfully extracted
#         if has_landmarks:
#             cmd += ['--landmarks', landmarks_csv]
            
#         if has_odom:
#             cmd += ['--odom', odom_csv]
            
#         run(cmd, "Step 2/4: SLAM Pipeline Metrics (ATE, RPE, 3σ + Landmarks)")
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

#!/usr/bin/env python3
#!/usr/bin/env python3
"""
run_all.py
==========
Master evaluation orchestrator for SLAM, Relocalization, and Global Localization.
"""

import argparse
import os
import subprocess
import sys

def run(cmd, description):
    print(f"\n{'='*60}\n  {description}\n{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  WARNING: step exited with code {result.returncode}")
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag',          default=None, help='Path to ROS2 bag')
    parser.add_argument('--csv_dir',      default=None, help='Directory with CSVs')
    parser.add_argument('--out',          required=True, help='Root results directory')
    parser.add_argument('--skip_extract', action='store_true')
    parser.add_argument('--rpe_window',   type=float, default=5.0)
    parser.add_argument('--error_threshold', type=float, default=2.0)
    parser.add_argument('--ekf_no_reloc', default=None, help='Path to the original EKF CSV (without reloc) from another directory')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = args.csv_dir if args.csv_dir else os.path.join(args.out, 'csv')
    os.makedirs(args.out, exist_ok=True); os.makedirs(csv_dir, exist_ok=True)

    # 1. Extraction
    if not args.skip_extract and args.bag:
        run([sys.executable, os.path.join(script_dir, 'extract_bag_ldm.py'), 
             '--bag', args.bag, '--out', csv_dir], "Step 1: Extracting Bag to CSVs")

    # Path setup
    ekf_csv   = os.path.join(csv_dir, 'ekf_poses.csv')
    gt_csv    = os.path.join(csv_dir, 'ground_truth.csv')
    lms_csv   = os.path.join(csv_dir, 'ekf_landmarks.csv')
    odom_csv  = os.path.join(csv_dir, 'wheel_odom.csv')
    global_csv= os.path.join(csv_dir, 'global_poses.csv')
    stats_csv = os.path.join(csv_dir, 'matcher_stats.csv')
    no_reloc_csv = os.path.join(args.ekf_no_reloc, 'ekf_poses.csv') if args.ekf_no_reloc else None

    # 2. SLAM Metrics (with Z-Trajectory)
    if os.path.exists(ekf_csv) and os.path.exists(gt_csv):
        cmd = [sys.executable, os.path.join(script_dir, 'eval_slam_ldm.py'), 
               '--ekf', ekf_csv, '--gt', gt_csv, '--out', os.path.join(args.out, 'slam'),
               '--rpe_window', str(args.rpe_window), '--global_poses', global_csv]
        if os.path.exists(lms_csv): cmd += ['--landmarks', lms_csv]
        if os.path.exists(odom_csv): cmd += ['--odom', odom_csv]
        if args.ekf_no_reloc:
            cmd += ['--ekf_no_reloc', no_reloc_csv]
        run(cmd, "Step 2: SLAM metrics and Full 3D Plotting")

    # 3. Relocalization Metrics
    if os.path.exists(global_csv) and os.path.exists(gt_csv) and os.path.exists(odom_csv):
        run([sys.executable, os.path.join(script_dir, 'eval_relocalization.py'),
             '--global_poses', global_csv, '--gt', gt_csv, '--odom', odom_csv,
             '--out', os.path.join(args.out, 'relocalization'),
             '--error_threshold', str(args.error_threshold)], "Step 3: Sequential Relocalization Metrics")

    # 4. Global Localization Metrics
    if os.path.exists(global_csv) and os.path.exists(gt_csv):
        cmd = [sys.executable, os.path.join(script_dir, 'eval_global_loc.py'),
               '--global_poses', global_csv, '--gt', gt_csv,
               '--out', os.path.join(args.out, 'global_loc')]
        if os.path.exists(stats_csv): cmd += ['--stats', stats_csv]
        run(cmd, "Step 4: Global Localization Metrics")

    print(f"\nAll evaluations complete. Results saved in: {args.out}")

if __name__ == '__main__': main()