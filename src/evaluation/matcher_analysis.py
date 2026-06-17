# pip3 install pandas matplotlib seaborn
#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    mars_yard = 'MYCNES'
    csv_path = os.path.expanduser(f'/home/ws/src/matcher_csvs/{mars_yard}.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find {csv_path}")
        print("Make sure you run your ROS node first to generate the data!")
        return

    print("📊 Loading RANSAC performance data...")
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print("⚠️ The CSV is empty.")
        return

    # --- DATA PREPARATION ---
    # Create an Inlier Ratio percentage
    df['inlier_ratio'] = (df['inlier_count'] / df['local_map_size']) * 100.0
    
    # Filter out any catastrophic failures where local map was 0
    df = df[df['local_map_size'] > 0].copy()
    
    # Calculate time elapsed since the start of the bag
    df['elapsed_time'] = df['timestamp_sec'] - df['timestamp_sec'].iloc[0]

    # --- TERMINAL SUMMARY STATISTICS ---
    print("\n" + "="*40)
    print("🚀 RANSAC PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Total Matches Run:    {len(df)}")
    print(f"Max Local Map Size:   {df['local_map_size'].max()} rocks")
    print(f"Early Exit Rate:      {(df['early_exit_triggered'].mean() * 100):.1f}%")
    print(f"Average Inlier Ratio: {df['inlier_ratio'].mean():.1f}%")
    print("-" * 40)
    print(f"Average Compute Time: {df['compute_time_ms'].mean():.2f} ms")
    print(f"Max Compute Time:     {df['compute_time_ms'].max():.2f} ms")
    print(f"99th Percentile Time: {df['compute_time_ms'].quantile(0.99):.2f} ms")
    print("="*40 + "\n")

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Global RANSAC Relocalization Performance', fontsize=16, fontweight='bold')

    # 1. Scalability: Compute Time vs. Local Map Size
    ax = axes[0, 0]
    sns.scatterplot(data=df, x='local_map_size', y='compute_time_ms', hue='status', 
                    palette={'SUCCESS': 'blue', 'FAILED': 'red'}, alpha=0.6, ax=ax)
    
    # Draw the 100ms budget line (10Hz)
    ax.axhline(y=100.0, color='r', linestyle='--', label='10Hz Budget (100ms)')
    # Draw a 15ms "Safe" threshold
    ax.axhline(y=15.0, color='orange', linestyle=':', label='15% CPU Core Limit (15ms)')
    
    ax.set_title('1. Scalability (Compute Time vs. Map Size)')
    ax.set_xlabel('Local Map Size (Number of Rocks)')
    ax.set_ylabel('Compute Time [ms]')
    ax.legend(loc='upper left')

    # 2. Efficiency: Iterations Used (Early Exit Proof)
    ax = axes[0, 1]
    sns.histplot(data=df, x='iterations_used', hue='early_exit_triggered', 
                 multiple="stack", bins=30, palette={True: 'green', False: 'gray'}, ax=ax)
    ax.set_title('2. Efficiency (RANSAC Iterations Used)')
    ax.set_xlabel('Iterations Executed')
    ax.set_ylabel('Frequency')
    # Custom legend
    handles = [plt.Rectangle((0,0),1,1, color='green', alpha=0.7), 
               plt.Rectangle((0,0),1,1, color='gray', alpha=0.7)]
    ax.legend(handles, ['Early Exit (Saved CPU)', 'Max Iterations (Worst Case)'])

    # 3. CPU Budget Check: Distribution of Compute Times
    ax = axes[1, 0]
    sns.histplot(data=df, x='compute_time_ms', kde=True, color='purple', bins=40, ax=ax)
    ax.set_title('3. Real-Time Viability (Compute Time Distribution)')
    ax.set_xlabel('Compute Time [ms]')
    ax.set_ylabel('Frequency')

    # 4. Robustness: Inlier Ratio over Time
    ax = axes[1, 1]
    sns.lineplot(data=df, x='elapsed_time', y='inlier_ratio', color='teal', alpha=0.8, ax=ax)
    sns.scatterplot(data=df, x='elapsed_time', y='inlier_ratio', hue='status', 
                    palette={'SUCCESS': 'blue', 'FAILED': 'red'}, alpha=0.5, s=20, ax=ax)
    
    ax.axhline(y=20.0, color='red', linestyle='--', alpha=0.5, label='20% Minimum Threshold')
    ax.set_title('4. Robustness (Inlier Ratio over Mission Time)')
    ax.set_xlabel('Mission Time [s]')
    ax.set_ylabel('Inliers / Total Local Rocks [%]')
    ax.legend(loc='lower right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save and Show
    out_file = os.path.expanduser('~/ransac_dashboard.png')
    plt.savefig(out_file, dpi=200)
    print(f"📈 Dashboard saved to {out_file}")
    plt.show()

if __name__ == '__main__':
    main()