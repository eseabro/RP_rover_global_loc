#!/usr/bin/env python3
"""
compute_timing.py
=================
Estimates computational overhead and correction statistics
from pre-extracted CSV files.

Usage:
    python3 compute_timing.py \
        --with_dir    /home/ws/results/MY2021/csv \
        --without_dir /home/ws/results/MY2021_ekf_only/csv
"""

import argparse
import os
import numpy as np
import pandas as pd


def analyze_csv_dir(csv_dir: str, label: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  CSV dir: {csv_dir}")
    print(f"{'='*55}")

    results = {}

    # ── EKF pose timestamps ──────────────────────────────────
    ekf_path = os.path.join(csv_dir, 'ekf_poses.csv')
    if os.path.exists(ekf_path):
        ekf = pd.read_csv(ekf_path).sort_values('timestamp')
        ekf_ts  = ekf['timestamp'].values
        ekf_dts = np.diff(ekf_ts)
        duration = ekf_ts[-1] - ekf_ts[0]
        ekf_rate = 1.0 / np.mean(ekf_dts)

        print(f"  EKF pose messages:        {len(ekf_ts)}")
        print(f"  Duration [s]:             {duration:.1f}")
        print(f"  Mean EKF publish rate:    {ekf_rate:.2f} Hz")
        print(f"  Mean EKF period [ms]:     {np.mean(ekf_dts)*1000:.1f}")
        print(f"  Std  EKF period [ms]:     {np.std(ekf_dts)*1000:.1f}")
        print(f"  Max  EKF period [ms]:     {np.max(ekf_dts)*1000:.1f}")

        results['ekf_rate_hz'] = ekf_rate
        results['ekf_period_mean_ms'] = np.mean(ekf_dts) * 1000
        results['ekf_period_std_ms']  = np.std(ekf_dts)  * 1000
        results['ekf_period_max_ms']  = np.max(ekf_dts)  * 1000
        results['duration'] = duration
        results['n_ekf'] = len(ekf_ts)
    else:
        print(f"  WARNING: ekf_poses.csv not found")
        results['ekf_rate_hz'] = None

    # ── Rock detections ──────────────────────────────────────
    # Try common names
    for name in ['rock_measurements.csv', 'rock_measurements_raw.csv']:
        rock_path = os.path.join(csv_dir, name)
        if os.path.exists(rock_path):
            rock = pd.read_csv(rock_path).sort_values('timestamp')
            rock_ts   = rock['timestamp'].values
            rock_rate = 1.0 / np.mean(np.diff(rock_ts))
            print(f"\n  Rock detection messages:  {len(rock_ts)}")
            print(f"  Mean detection rate:      {rock_rate:.2f} Hz")
            results['rock_rate_hz'] = rock_rate
            results['n_rocks'] = len(rock_ts)
            break

    # ── Global corrections ───────────────────────────────────
    for name in ['global_poses.csv', 'rock_global_pose.csv']:
        gp_path = os.path.join(csv_dir, name)
        if os.path.exists(gp_path):
            gp = pd.read_csv(gp_path).sort_values('timestamp')
            gp_ts = gp['timestamp'].values

            if len(gp_ts) > 1:
                intervals = np.diff(gp_ts)
                corr_rate = len(gp_ts) / results.get('duration', 1.0)

                print(f"\n  Global corrections:       {len(gp_ts)}")
                print(f"  Correction rate:          {corr_rate:.4f} Hz")
                print(f"  Mean interval [s]:        {np.mean(intervals):.1f}")
                print(f"  Std  interval [s]:        {np.std(intervals):.1f}")
                print(f"  Min  interval [s]:        {np.min(intervals):.1f}")
                print(f"  Max  interval [s]:        {np.max(intervals):.1f}")

                results['n_corrections']       = len(gp_ts)
                results['correction_rate_hz']  = corr_rate
                results['interval_mean_s']     = np.mean(intervals)
                results['interval_std_s']      = np.std(intervals)
                results['interval_min_s']      = np.min(intervals)
                results['interval_max_s']      = np.max(intervals)
            else:
                print(f"\n  Global corrections:       {len(gp_ts)} (too few to analyze)")
                results['n_corrections'] = len(gp_ts)
            break
    else:
        print(f"\n  Global corrections:       0 (EKF only or not recorded)")
        results['n_corrections'] = 0

    results['label'] = label
    return results


def print_comparison(r1: dict, r2: dict):
    print(f"\n{'='*55}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Metric':<35} {'With Matcher':>12} {'EKF Only':>12}")
    print(f"  {'-'*59}")

    def row(name, key, fmt='.2f', unit=''):
        v1 = r1.get(key)
        v2 = r2.get(key)
        s1 = f"{v1:{fmt}}{unit}" if v1 is not None else 'N/A'
        s2 = f"{v2:{fmt}}{unit}" if v2 is not None else 'N/A'
        print(f"  {name:<35} {s1:>12} {s2:>12}")

    row('Duration [s]',             'duration',           '.1f')
    row('EKF publish rate [Hz]',    'ekf_rate_hz',        '.2f')
    row('Mean EKF period [ms]',     'ekf_period_mean_ms', '.1f')
    row('Max EKF period [ms]',      'ekf_period_max_ms',  '.1f')
    row('Global corrections',       'n_corrections',      '.0f')
    row('Correction rate [Hz]',     'correction_rate_hz', '.4f')
    row('Mean correction interval', 'interval_mean_s',    '.1f', 's')

    # Rate overhead
    if r1.get('ekf_rate_hz') and r2.get('ekf_rate_hz'):
        diff = r1['ekf_rate_hz'] - r2['ekf_rate_hz']
        print(f"\n  EKF rate difference: {diff:+.2f} Hz")
        if diff < 0:
            print(f"  → Matcher adds ~{abs(diff):.2f} Hz of overhead to the EKF loop")
        else:
            print(f"  → No measurable overhead from matcher")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_dir',    required=True,
                        help='CSV directory from run WITH global matching')
    parser.add_argument('--without_dir', required=True,
                        help='CSV directory from run WITHOUT global matching')
    args = parser.parse_args()

    r1 = analyze_csv_dir(args.with_dir,    'EKF + Global Matcher')
    r2 = analyze_csv_dir(args.without_dir, 'EKF Only')
    print_comparison(r1, r2)


if __name__ == '__main__':
    main()