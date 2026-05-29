import pandas as pd

def evaluate_metrics(file_path):
    # Load the CSV data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return

    # Total runs for percentage calculations
    total_runs = len(df)
    if total_runs == 0:
        print("The CSV is empty.")
        return

    # 1. Status & Stability Metrics
    success_count = len(df[df['status'] == 'SUCCESS'])
    success_rate = (success_count / total_runs) * 100
    
    # Handle string 'True'/'False' or boolean types for early_exit
    early_exits = df['early_exit_triggered'].astype(str).str.lower() == 'true'
    early_exit_count = early_exits.sum()

    # 2. Performance Metrics (Compute Time)
    avg_compute = df['compute_time_ms'].mean()
    max_compute = df['compute_time_ms'].max()
    min_compute = df['compute_time_ms'].min()

    # 3. Accuracy & Scale Metrics
    avg_rmse = df['rmse'].mean()
    avg_inliers = df['inlier_count'].mean()
    avg_map_size = df['local_map_size'].mean()

    # Print the evaluation
    print("====== EKF / SLAM METRICS EVALUATION ======")
    print(f"Total Logs Evaluated: {total_runs}")
    print(f"Success Rate:         {success_rate:.2f}% ({success_count}/{total_runs})")
    print(f"Early Exits:          {early_exit_count}")
    print("-------------------------------------------")
    print("⏱️  COMPUTE TIME (ms)")
    print(f"   Average:           {avg_compute:.2f}")
    print(f"   Min / Max:         {min_compute:.2f} / {max_compute:.2f}")
    print("-------------------------------------------")
    print("🎯 ACCURACY & SCALE")
    print(f"   Average RMSE:      {avg_rmse:.4f}")
    print(f"   Average Inliers:   {avg_inliers:.1f}")
    print(f"   Avg Map Size:      {avg_map_size:.1f} landmarks")
    print("===========================================")

if __name__ == "__main__":
    # Replace with the actual path to your CSV file
    NAME = '2022' # Change to '2020' if that is your default
    csv_file = f"/home/ws/src/matcher_csvs/MY{NAME}.csv" 
    evaluate_metrics(csv_file)