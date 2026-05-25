#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

with_csv    = '/home/ws/results/run_21_long/csv/ekf_poses.csv'
without_csv = '/home/ws/results/run_22_long_no/csv/ekf_poses.csv'
out_dir     = '/home/ws/results/'

df_with    = pd.read_csv(with_csv).sort_values('timestamp')
df_without = pd.read_csv(without_csv).sort_values('timestamp')

dts_with    = np.diff(df_with['timestamp'].values)    * 1000  # ms
dts_without = np.diff(df_without['timestamp'].values) * 1000  # ms

# Remove outliers beyond 500ms for clean histogram
dts_with_c    = dts_with[dts_with < 500]
dts_without_c = dts_without[dts_without < 500]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram comparison
axes[0].hist(dts_without_c, bins=80, alpha=0.6,
             color='steelblue', label=f'EKF Only (mean={np.mean(dts_without):.1f}ms)')
axes[0].hist(dts_with_c,    bins=80, alpha=0.6,
             color='tomato',    label=f'EKF+Matcher (mean={np.mean(dts_with):.1f}ms)')
axes[0].set_xlabel('EKF Period [ms]')
axes[0].set_ylabel('Count')
axes[0].set_title('EKF Update Period Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: period over time (shows when spikes occur)
t_with    = df_with['timestamp'].values[1:]
t_without = df_without['timestamp'].values[1:]
t_with    = t_with    - t_with[0]
t_without = t_without - t_without[0]

axes[1].plot(t_without, dts_without, color='steelblue',
             alpha=0.5, linewidth=0.5, label='EKF Only')
axes[1].plot(t_with,    dts_with,    color='tomato',
             alpha=0.5, linewidth=0.5, label='EKF+Matcher')
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('EKF Period [ms]')
axes[1].set_title('EKF Period Over Time')
axes[1].set_ylim(0, 500)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Print percentile comparison
print("\n=== EKF Period Percentiles [ms] ===")
print(f"{'Percentile':<15} {'EKF Only':>12} {'EKF+Matcher':>12}")
print("-" * 40)
for p in [50, 75, 90, 95, 99, 100]:
    v1 = np.percentile(dts_without, p)
    v2 = np.percentile(dts_with, p)
    print(f"  p{p:<13} {v1:>12.1f} {v2:>12.1f}")

plt.tight_layout()
path = os.path.join(out_dir, 'ekf_period_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {path}")