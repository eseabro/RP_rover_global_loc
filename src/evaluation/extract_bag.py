#!/usr/bin/env python3
"""
extract_bag.py
==============
Reads a ROS2 bag (sqlite3 OR mcap) and extracts relevant topics to CSV files.

Usage:
    python3 extract_bag.py --bag /path/to/bag --out /path/to/output_dir

Output CSVs:
    ekf_poses.csv       - EKF estimated poses
    ground_truth.csv    - Ground truth poses
    wheel_odom.csv      - Wheel odometry poses  (/cmd_odom auto-detected)
    global_poses.csv    - Global relocalization estimates
    matcher_stats.csv   - Matcher statistics (inliers, query time)
"""

import argparse
import os
import csv
import math


def get_yaw_from_quaternion(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {path}")


def pose_row_from_odometry(msg):
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z
    qx = msg.pose.pose.orientation.x
    qy = msg.pose.pose.orientation.y
    qz = msg.pose.pose.orientation.z
    qw = msg.pose.pose.orientation.w
    yaw = get_yaw_from_quaternion(qx, qy, qz, qw)
    cov = msg.pose.covariance
    return [t, x, y, z, yaw, cov[0], cov[7], cov[14], cov[35]]


def pose_row_from_pose_with_cov(msg):
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z
    qx = msg.pose.pose.orientation.x
    qy = msg.pose.pose.orientation.y
    qz = msg.pose.pose.orientation.z
    qw = msg.pose.pose.orientation.w
    yaw = get_yaw_from_quaternion(qx, qy, qz, qw)
    cov = msg.pose.covariance
    return [t, x, y, z, yaw, cov[0], cov[7], cov[14], cov[35]]


def detect_bag_format(bag_path):
    for f in os.listdir(bag_path):
        if f.endswith('.db3'):
            return 'sqlite3'
        if f.endswith('.mcap'):
            return 'mcap'
    raise FileNotFoundError(f"No .db3 or .mcap file found in {bag_path}")


def read_bag(bag_path, topics_of_interest):
    """Read messages using rosbag2_py API — works for both sqlite3 and mcap."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as e:
        raise ImportError(
            f"rosbag2_py not available: {e}\n"
            f"Make sure you sourced ROS2: source /opt/ros/jazzy/setup.bash"
        )

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='')
    converter_options = rosbag2_py.ConverterOptions('', '')

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    print("\nTopics found in bag:")
    for t in topic_types:
        print(f"  {t.name}  ({t.type})")

    storage_filter = rosbag2_py.StorageFilter(topics=topics_of_interest)
    reader.set_filter(storage_filter)

    result = {t: [] for t in topics_of_interest}

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in type_map and topic in result:
            try:
                msg = deserialize_message(data, get_message(type_map[topic]))
                result[topic].append(msg)
            except Exception:
                pass

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag',         required=True, help='ROS2 bag directory')
    parser.add_argument('--out',         required=True, help='Output CSV directory')
    parser.add_argument('--odom_topic',  default=None,
                        help='Override odometry topic (default: auto-detect /cmd_odom or /wheel_odom)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    fmt = detect_bag_format(args.bag)
    print(f"Bag format detected: {fmt}")
    print(f"Reading: {args.bag}")

    topics = [
        '/ekf/pose',
        '/ground_truth/odom',
        '/wheel_odom',
        '/cmd_odom',
        '/rock_global_pose',
        '/matcher/stats',
    ]

    messages = read_bag(args.bag, topics)

    POSE_HEADER = ['timestamp', 'x', 'y', 'z', 'yaw',
                   'cov_xx', 'cov_yy', 'cov_zz', 'cov_yaw']

    # ── EKF Poses ────────────────────────────────────────────────
    if messages['/ekf/pose']:
        rows = [pose_row_from_pose_with_cov(m) for m in messages['/ekf/pose']]
        write_csv(os.path.join(args.out, 'ekf_poses.csv'), POSE_HEADER, rows)
    else:
        print("  WARNING: /ekf/pose not in bag — run pipeline while replaying to generate this")

    # ── Ground Truth ─────────────────────────────────────────────
    if messages['/ground_truth/odom']:
        rows = [pose_row_from_odometry(m) for m in messages['/ground_truth/odom']]
        write_csv(os.path.join(args.out, 'ground_truth.csv'), POSE_HEADER, rows)
    else:
        print("  WARNING: /ground_truth/odom not in bag — ATE metrics unavailable")

    # ── Odometry (auto-detect /cmd_odom vs /wheel_odom) ──────────
    odom_topic = args.odom_topic
    if odom_topic is None:
        if messages['/cmd_odom']:
            odom_topic = '/cmd_odom'
        elif messages['/wheel_odom']:
            odom_topic = '/wheel_odom'

    if odom_topic and messages.get(odom_topic):
        rows = [pose_row_from_odometry(m) for m in messages[odom_topic]]
        write_csv(os.path.join(args.out, 'wheel_odom.csv'), POSE_HEADER, rows)
        print(f"  Odometry source: {odom_topic}")
    else:
        print("  WARNING: No odometry topic found")

    # ── Global Poses ─────────────────────────────────────────────
    if messages['/rock_global_pose']:
        rows = [pose_row_from_pose_with_cov(m) for m in messages['/rock_global_pose']]
        write_csv(os.path.join(args.out, 'global_poses.csv'), POSE_HEADER, rows)
    else:
        print("  WARNING: /rock_global_pose not in bag — relocalization metrics unavailable")

    # ── Matcher Stats ────────────────────────────────────────────
    if messages['/matcher/stats']:
        rows = []
        for m in messages['/matcher/stats']:
            d = m.data
            if len(d) >= 5:
                rows.append([d[0], d[1], d[2], d[3], d[4]])
        write_csv(os.path.join(args.out, 'matcher_stats.csv'),
                  ['timestamp', 'n_inliers', 'n_outliers', 'query_time_ms', 'db_size'],
                  rows)
    else:
        print("  NOTE: /matcher/stats not in bag — add stats publisher to matcher_node.py")

    print(f"\nExtraction complete. CSVs in: {args.out}")
    print("Next step:")
    print(f"  python3 run_all.py --csv_dir {args.out} --out <results_dir> --skip_extract")


if __name__ == '__main__':
    main()