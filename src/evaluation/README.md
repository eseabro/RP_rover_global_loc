# Evaluation Scripts — Mars Rover SLAM Pipeline

## Structure

```
evaluation/
├── extract_bag.py          # Step 1: ROS2 bag → CSV files
├── eval_slam.py            # Step 2: Full SLAM pipeline metrics
├── eval_relocalization.py  # Step 3: Sequential relocalization metrics
├── eval_global_loc.py      # Step 4: Global localization metrics
├── run_all.py              # Master script — runs all steps
└── README.md
```

## Dependencies

```bash
pip install numpy pandas matplotlib scipy --break-system-packages
# ROS2 packages (already available in dev container):
# rclpy, rosidl_runtime_py
```

## Workflow

### Full pipeline (bag → results):
```bash
python3 run_all.py \
    --bag /home/ws/rosbag2_2026_03_10-11_38_47 \
    --out /home/ws/results/run_01/
```

### Re-run evaluation on already-extracted CSVs:
```bash
python3 run_all.py \
    --csv_dir /home/ws/results/run_01/csv \
    --out /home/ws/results/run_01/ \
    --skip_extract
```

### Run individual steps:
```bash
# Extract only
python3 extract_bag.py --bag /path/to/bag --out /path/to/csv/

# SLAM metrics only
python3 eval_slam.py --ekf ekf_poses.csv --gt ground_truth.csv \
                     --odom wheel_odom.csv --out results/slam/

# Relocalization metrics only
python3 eval_relocalization.py --global_poses global_poses.csv \
                               --gt ground_truth.csv \
                               --odom wheel_odom.csv \
                               --out results/reloc/

# Global localization metrics only
python3 eval_global_loc.py --global_poses global_poses.csv \
                           --gt ground_truth.csv \
                           --out results/global_loc/
```

## Required topics in the bag

| Topic                  | Required for          |
|------------------------|-----------------------|
| `/ekf/pose`            | All SLAM metrics      |
| `/ground_truth/odom`   | All metrics           |
| `/wheel_odom`          | ATE comparison, RPE   |
| `/rock_global_pose`    | Relocalization metrics|
| `/matcher/stats`       | Inlier/query metrics  |

**Ask Emma to record a bag with all of these, including a loop trajectory.**

## Outputs

### SLAM metrics (`results/slam/`)
- `trajectories.png` — GT vs EKF vs odometry path
- `ate_over_time.png` — position error over time
- `sigma_consistency.png` — error vs 3σ covariance bound
- `covariance_over_time.png` — EKF uncertainty evolution
- `slam_summary.csv` — all numeric metrics

### Relocalization metrics (`results/relocalization/`)
- `reloc_error_vs_distance.png` — error at each reloc event
- `success_rate_vs_distance.png` — success rate in distance bins
- `reloc_on_trajectory.png` — reloc events marked on trajectory
- `reloc_summary.csv` — all numeric metrics

### Global localization metrics (`results/global_loc/`)
- `error_distribution.png` — histogram of position and yaw errors
- `error_scatter.png` — X/Y error scatter with σ ellipses
- `inlier_stats.png` — inlier count and query time (if stats available)
- `global_loc_summary.csv` — all numeric metrics

## To enable inlier/query time metrics

Add to `matcher_node.py` `__init__`:
```python
from std_msgs.msg import Float32MultiArray
self.stats_pub = self.create_publisher(Float32MultiArray, '/matcher/stats', 10)
```

After each match:
```python
import time
t_start = time.time()
# ... matching code ...
query_time_ms = (time.time() - t_start) * 1000

stats_msg = Float32MultiArray()
stats_msg.data = [float(n_inliers), float(n_outliers),
                  float(query_time_ms), float(db_size)]
self.stats_pub.publish(stats_msg)
```
