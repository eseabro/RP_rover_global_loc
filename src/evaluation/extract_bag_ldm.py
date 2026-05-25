#!/usr/bin/env python3
import argparse
import os
import csv
import math
import numpy as np

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

# --- NEW: Process Landmarks from MarkerArray ---
def landmark_rows_from_marker_array(msg):
    """
    Returns a list of rows, one for each marker in the array.
    """
    t = msg.markers[0].header.stamp.sec + msg.markers[0].header.stamp.nanosec * 1e-9 if msg.markers else 0.0
    rows = []
    for m in msg.markers:
        # We extract: ID, X, Y, Z, and Dimensions (Scale)
        rows.append([
            t,
            m.id,
            m.pose.position.x,
            m.pose.position.y,
            m.pose.position.z,
            m.scale.x,  # Width/Diameter
            m.scale.y,  # Depth/Diameter
            m.scale.z,  # Height
            m.color.a   # Alpha can indicate if it's archived (0.4) or active (0.9)
        ])
    return rows

def detect_bag_format(bag_path):
    for f in os.listdir(bag_path):
        if f.endswith('.db3'): return 'sqlite3'
        if f.endswith('.mcap'): return 'mcap'
    raise FileNotFoundError(f"No .db3 or .mcap file found in {bag_path}")

def read_bag(bag_path, topics_of_interest):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as e:
        raise ImportError("rosbag2_py not available. Source ROS2 first.")

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    
    storage_filter = rosbag2_py.StorageFilter(topics=topics_of_interest)
    reader.set_filter(storage_filter)

    result = {t: [] for t in topics_of_interest}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in type_map and topic in result:
            msg = deserialize_message(data, get_message(type_map[topic]))
            result[topic].append(msg)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True, help='ROS2 bag directory')
    parser.add_argument('--out', required=True, help='Output CSV directory')
    parser.add_argument('--odom_topic', default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fmt = detect_bag_format(args.bag)

    topics = [
        '/ekf/pose',
        '/ekf/landmarks',   # Added
        '/ground_truth/odom',
        '/wheel_odom',
        '/cmd_odom',
        '/ekf/odom',
        '/rock_global_pose',
        '/matcher/stats',
    ]

    messages = read_bag(args.bag, topics)

    POSE_HEADER = ['timestamp', 'x', 'y', 'z', 'yaw', 'cov_xx', 'cov_yy', 'cov_zz', 'cov_yaw']
    LANDMARK_HEADER = ['timestamp', 'id', 'x', 'y', 'z', 'dim_x', 'dim_y', 'dim_z', 'alpha']

    # ── EKF Poses ────────────────────────────────────────────────
    if messages['/ekf/pose']:
        rows = [pose_row_from_pose_with_cov(m) for m in messages['/ekf/pose']]
        write_csv(os.path.join(args.out, 'ekf_poses.csv'), POSE_HEADER, rows)

    # ── NEW: EKF Landmarks ───────────────────────────────────────
    if messages['/ekf/landmarks']:
        all_landmark_rows = []
        for msg in messages['/ekf/landmarks']:
            all_landmark_rows.extend(landmark_rows_from_marker_array(msg))
        write_csv(os.path.join(args.out, 'ekf_landmarks.csv'), LANDMARK_HEADER, all_landmark_rows)
    else:
        print("  WARNING: /ekf/landmarks not in bag")

    # ── Ground Truth ─────────────────────────────────────────────
    if messages['/ground_truth/odom']:
        rows = [pose_row_from_odometry(m) for m in messages['/ground_truth/odom']]
        write_csv(os.path.join(args.out, 'ground_truth.csv'), POSE_HEADER, rows)

    # ── Odometry ─────────────────────────────────────────────────
    odom_topic = args.odom_topic or ('/ekf/odom' if messages['/ekf/odom'] else '/cmd_odom')
    if odom_topic and messages.get(odom_topic):
        rows = [pose_row_from_odometry(m) for m in messages[odom_topic]]
        write_csv(os.path.join(args.out, 'wheel_odom.csv'), POSE_HEADER, rows)

    # ── Global Poses ─────────────────────────────────────────────
    if messages['/rock_global_pose']:
        rows = [pose_row_from_pose_with_cov(m) for m in messages['/rock_global_pose']]
        write_csv(os.path.join(args.out, 'global_poses.csv'), POSE_HEADER, rows)

    # ── Matcher Stats ────────────────────────────────────────────
    if messages['/matcher/stats']:
        stats_rows = [[m.data[0], m.data[1], m.data[2], m.data[3], m.data[4]] for m in messages['/matcher/stats'] if len(m.data) >= 5]
        write_csv(os.path.join(args.out, 'matcher_stats.csv'), 
                  ['timestamp', 'n_inliers', 'n_outliers', 'query_time_ms', 'db_size'], stats_rows)

    print(f"\nExtraction complete. CSVs in: {args.out}")

if __name__ == '__main__':
    main()