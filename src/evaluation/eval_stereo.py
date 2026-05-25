#!/usr/bin/env python3
"""
eval_stereo.py
==============
Offline evaluation of the stereo 3D rock localization pipeline.
Uses Gazebo LiDAR (/scan/points) as ground truth depth reference.
Runs YOLO offline to generate masks for visualization.

Produces:
  1. disparity_examples.png  - left image + SGBM disparity side by side (N frames)
  2. detection_examples.png  - left image + YOLO mask + centroid overlay (N frames)
  3. depth_error.csv         - per-observation depth error table (stereo vs LiDAR)
  4. stereo_summary.csv      - aggregate metrics for thesis table

Bag must contain:
  /camera/left/image_rect_color   (sensor_msgs/Image)
  /camera/right/image_rect_color  (sensor_msgs/Image)
  /camera/left/camera_info        (sensor_msgs/CameraInfo)
  /rock_measurements_raw          (visualization_msgs/MarkerArray)
  /scan/points                    (sensor_msgs/PointCloud2)
  /tf  /tf_static

Usage:
    python3 eval_stereo.py \
        --bag   /home/ws/stereo_eval_lidar \
        --model /home/ws/ROS_ws/src/custom_slam/models/rock_seg.pt \
        --out   /home/ws/results/stereo_eval/ \
        --n_examples 4
"""

import argparse
import os
import sys
import struct
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import csv


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class Frame:
    timestamp:   float
    left_img:    Optional[np.ndarray] = None
    right_img:   Optional[np.ndarray] = None
    camera_info: Optional[dict]       = None  # fx,fy,cx,cy,baseline
    rock_obs:    List[dict]           = field(default_factory=list)
    lidar_pts:   Optional[np.ndarray] = None  # Nx3 in sensor frame


# ─────────────────────────────────────────────────────────────
# Bag reader
# ─────────────────────────────────────────────────────────────

def read_bag(bag_path: str) -> List[Frame]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        raise ImportError("Source ROS2 first: source /opt/ros/jazzy/setup.bash")

    storage_options   = rosbag2_py.StorageOptions(uri=bag_path, storage_id='')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map    = {t.name: t.type for t in topic_types}

    print("\nTopics found in bag:")
    for t in topic_types:
        print(f"  {t.name}")

    TOPICS = [
        '/camera/left/image_rect_color',
        '/camera/right/image_rect_color',
        '/camera/left/camera_info',
        '/rock_measurements_raw',
        '/scan/points',
    ]
    # Only filter topics that actually exist
    available = [t for t in TOPICS if t in type_map]
    print(f"\nUsing topics: {available}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=available))

    left_buf  = {}
    right_buf = {}
    info_buf  = {}
    lidar_buf = {}
    rock_msgs = []

    def decode_image(data, msg_type):
        msg = deserialize_message(data, get_message(msg_type))
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        h, w = msg.height, msg.width
        enc  = msg.encoding.lower()
        if 'bgr8' in enc:
            img = arr.reshape(h, w, 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif 'rgb8' in enc:
            img = arr.reshape(h, w, 3)
        elif 'mono8' in enc or '8uc1' in enc:
            img = arr.reshape(h, w)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = arr.reshape(h, w, -1)[:, :, :3]
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return ts, img

    def decode_pointcloud2(data, msg_type):
        """Decode PointCloud2 to Nx3 numpy array [x,y,z]."""
        msg = deserialize_message(data, get_message(msg_type))
        ts  = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Find field offsets
        fields = {f.name: f.offset for f in msg.fields}
        if 'x' not in fields:
            return ts, None

        pts = []
        point_step = msg.point_step
        raw = bytes(msg.data)
        for i in range(msg.width * msg.height):
            base = i * point_step
            x = struct.unpack_from('f', raw, base + fields['x'])[0]
            y = struct.unpack_from('f', raw, base + fields['y'])[0]
            z = struct.unpack_from('f', raw, base + fields['z'])[0]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                pts.append([x, y, z])

        return ts, np.array(pts) if pts else None

    while reader.has_next():
        topic, data, bag_ts = reader.read_next()
        t_sec = bag_ts * 1e-9

        if topic == '/camera/left/image_rect_color':
            ts, img = decode_image(data, type_map[topic])
            left_buf[ts] = img

        elif topic == '/camera/right/image_rect_color':
            ts, img = decode_image(data, type_map[topic])
            right_buf[ts] = img

        elif topic == '/camera/left/camera_info':
            msg = deserialize_message(data, get_message(type_map[topic]))
            ts  = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            K   = msg.k
            P   = msg.p
            fx  = float(K[0]); fy = float(K[4])
            cx  = float(K[2]); cy = float(K[5])
            baseline = abs(float(P[3])) / fx if fx > 0 else 0.1
            info_buf[ts] = dict(fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline)

        elif topic == '/scan/points':
            ts, pts = decode_pointcloud2(data, type_map[topic])
            if pts is not None:
                lidar_buf[ts] = pts

        elif topic == '/rock_measurements_raw':
            msg = deserialize_message(data, get_message(type_map[topic]))
            ts  = (msg.markers[0].header.stamp.sec +
                   msg.markers[0].header.stamp.nanosec * 1e-9
                   if msg.markers else t_sec)
            rocks = []
            for marker in msg.markers:
                if marker.action == 3:
                    continue
                if marker.points:
                    pts_arr = np.array([[p.x, p.y, p.z] for p in marker.points])
                    centroid = np.median(pts_arr, axis=0)
                    rocks.append({'centroid': centroid,
                                  'points':   pts_arr,
                                  'confidence': float(marker.color.a)})
                else:
                    c = np.array([marker.pose.position.x,
                                  marker.pose.position.y,
                                  marker.pose.position.z])
                    rocks.append({'centroid': c, 'points': None,
                                  'confidence': float(marker.color.a)})
            if rocks:
                rock_msgs.append((ts, rocks))

    print(f"\n  Rock frames:   {len(rock_msgs)}")
    print(f"  Left images:   {len(left_buf)}")
    print(f"  Right images:  {len(right_buf)}")
    print(f"  LiDAR frames:  {len(lidar_buf)}")

    def nearest(buf, ts, max_dt=0.5):
        if not buf:
            return None
        times = np.array(list(buf.keys()))
        idx   = np.argmin(np.abs(times - ts))
        return buf[times[idx]] if abs(times[idx] - ts) <= max_dt else None

    frames = []
    for ts, rocks in rock_msgs:
        f             = Frame(timestamp=ts, rock_obs=rocks)
        f.left_img    = nearest(left_buf,  ts)
        f.right_img   = nearest(right_buf, ts)
        f.camera_info = nearest(info_buf,  ts, max_dt=5.0)
        f.lidar_pts   = nearest(lidar_buf, ts)
        frames.append(f)

    print(f"  Synchronized frames: {len(frames)}")
    return frames


# ─────────────────────────────────────────────────────────────
# SGBM disparity
# ─────────────────────────────────────────────────────────────

def compute_disparity(left_img: np.ndarray,
                      right_img: np.ndarray) -> np.ndarray:
    left_g  = cv2.equalizeHist(cv2.cvtColor(left_img,  cv2.COLOR_RGB2GRAY))
    right_g = cv2.equalizeHist(cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY))
    stereo  = cv2.StereoSGBM_create(
        minDisparity   = -16,
        numDisparities = 128,
        blockSize      = 5,
        P1             = 8  * 3 * 9**2,
        P2             = 32 * 3 * 9**2,
        disp12MaxDiff  = 1,
        uniquenessRatio    = 15,
        speckleWindowSize  = 100,
        speckleRange       = 32,
        preFilterCap       = 63,
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp = stereo.compute(left_g, right_g).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp


# ─────────────────────────────────────────────────────────────
# LiDAR depth for a rock
# ─────────────────────────────────────────────────────────────

def lidar_depth_for_rock(lidar_pts: np.ndarray,
                          centroid_cam: np.ndarray,
                          search_radius_m: float = 0.8) -> Optional[float]:
    """
    Confirmed frames from debug output:
      camera optical: x=right, y=down, z=forward
      base_scan:      x=backward, y=left, z=up
    Transform: lid_x = -cam_z, lid_y = -cam_x
    Depth = -lid_x = cam_z (forward distance in metres)
    """
    if lidar_pts is None or len(lidar_pts) == 0:
        return None
    cx_cam, cy_cam, cz_cam = centroid_cam
    cx_lid = -cz_cam
    cy_lid = -cx_cam
    dxy = np.sqrt((lidar_pts[:, 0] - cx_lid)**2 +
                  (lidar_pts[:, 1] - cy_lid)**2)
    nearby = lidar_pts[dxy < search_radius_m]
    if len(nearby) < 3:
        return None
    return float(np.median(-nearby[:, 0]))


# ─────────────────────────────────────────────────────────────
# YOLO inference (offline)
# ─────────────────────────────────────────────────────────────

def run_yolo(model, img_rgb: np.ndarray, conf: float = 0.25):
    """Run YOLO on an RGB image, return list of {mask, box, conf}."""
    import cv2
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results  = model(img_bgr, conf=conf, verbose=False)[0]
    detections = []
    if results.masks is not None and results.boxes is not None:
        h, w = img_rgb.shape[:2]
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) != 1:   # rock only
                continue
            mask_raw = results.masks.data[i].cpu().numpy()
            if mask_raw.shape != (h, w):
                mask_raw = cv2.resize(mask_raw, (w, h),
                                      interpolation=cv2.INTER_NEAREST)
            detections.append({
                'mask': mask_raw > 0.5,
                'box':  box.xyxy[0].cpu().numpy().astype(int),
                'conf': float(box.conf[0]),
            })
    return detections


# ─────────────────────────────────────────────────────────────
# Plot 1: Disparity examples
# ─────────────────────────────────────────────────────────────

def plot_depth_examples(frames: List[Frame], out_dir: str):
    """Single frame: left image + depth map side by side. No title."""
    valid = [f for f in frames
             if f.left_img is not None and f.right_img is not None
             and f.camera_info is not None]
    if not valid:
        print("  WARNING: No stereo pairs found for depth plot")
        return

    # Pick the frame with the most rock observations
    frame = max(valid, key=lambda f: len(f.rock_obs))

    h, w = frame.left_img.shape[:2]
    depth = np.full((h, w), np.nan)
    fx = frame.camera_info['fx']
    fy = frame.camera_info['fy']
    cx = frame.camera_info['cx']
    cy = frame.camera_info['cy']
    for rock in frame.rock_obs:
        if rock['points'] is None:
            continue
        pts = rock['points']
        pts = pts[pts[:, 2] > 0.01]
        us = np.clip((fx * pts[:, 0] / pts[:, 2] + cx).astype(int), 0, w-1)
        vs = np.clip((fy * pts[:, 1] / pts[:, 2] + cy).astype(int), 0, h-1)
        for u, v, z in zip(us, vs, pts[:, 2]):
            depth[v, u] = z

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(frame.left_img)
    axes[0].set_title('Left Camera Image', fontsize=11)
    axes[0].axis('off')

    vmin = 0.0
    vmax = float(np.nanpercentile(depth, 95))
    im   = axes[1].imshow(depth, cmap='plasma_r', vmin=vmin, vmax=vmax)
    axes[1].set_title('Stereo Depth Map [m]', fontsize=11)
    axes[1].axis('off')
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth [m]', fontsize=10)

    # Mark stereo centroids on depth map
    fx2 = frame.camera_info['fx']
    fy2 = frame.camera_info['fy']
    cx2 = frame.camera_info['cx']
    cy2 = frame.camera_info['cy']
    h, w = frame.left_img.shape[:2]
    for rock in frame.rock_obs:
        c = rock['centroid']
        if c[2] > 0:
            u = int(fx2 * c[0] / c[2] + cx2)
            v = int(fy2 * c[1] / c[2] + cy2)
            if 0 <= u < w and 0 <= v < h:
                axes[1].plot(u, v, 'w+', markersize=14, markeredgewidth=2)
                axes[1].annotate(
                    f'{c[2]:.1f} m',
                    xy=(u, v), xytext=(u + 15, v - 15),
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='black', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', color='white'))

    plt.tight_layout()
    path = os.path.join(out_dir, 'depth_example.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: depth_example.png")


# ─────────────────────────────────────────────────────────────
# Plot 2: Detection + mask + centroid overlay
# ─────────────────────────────────────────────────────────────

def plot_detection_examples(frames: List[Frame], model,
                             n: int, out_dir: str, conf: float = 0.25):
    valid = [f for f in frames
             if f.left_img is not None and f.rock_obs
             and f.camera_info is not None]
    if not valid:
        print("  WARNING: No frames with detections for overlay plot")
        return

    # Pick single best frame (most rocks detected)
    frame   = max(valid, key=lambda f: len(f.rock_obs))
    sampled = [frame]
    n_cols  = 1

    fig, axes = plt.subplots(1, 1, figsize=(10, 7))
    axes = [[axes]]  # keep indexing consistent
    # No figure title per thesis requirements

    PALETTE = plt.cm.Set1(np.linspace(0, 0.9, 9))

    for col, frame in enumerate(sampled):
        ax  = axes[0][col]
        img = frame.left_img.copy()

        # Run YOLO offline for masks
        detections = run_yolo(model, img, conf=conf)

        overlay = img.astype(np.float32)
        h, w    = img.shape[:2]

        for di, det in enumerate(detections):
            color_f = PALETTE[di % len(PALETTE)]
            color_u = (np.array(color_f[:3]) * 255).astype(np.uint8)

            # Semi-transparent mask
            mask = det['mask']
            overlay[mask] = 0.45 * overlay[mask] + 0.55 * color_u

            # Bounding box
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(overlay.astype(np.uint8),
                           (x1, y1), (x2, y2),
                           color_u.tolist(), 2)

        ax.imshow(overlay.astype(np.uint8))

        # Project stereo centroids
        fx = frame.camera_info['fx']
        fy = frame.camera_info['fy']
        cx = frame.camera_info['cx']
        cy = frame.camera_info['cy']

        for ri, rock in enumerate(frame.rock_obs):
            color_f = PALETTE[ri % len(PALETTE)]
            c       = rock['centroid']
            if c[2] <= 0:
                continue
            u = int(fx * c[0] / c[2] + cx)
            v = int(fy * c[1] / c[2] + cy)
            if 0 <= u < w and 0 <= v < h:
                ax.plot(u, v, 'o',
                        color=color_f, markersize=10,
                        markeredgecolor='white', markeredgewidth=1.5,
                        zorder=5)
                ax.annotate(
                    f'z = {c[2]:.2f} m',
                    xy=(u, v), xytext=(u + 14, v - 14),
                    fontsize=8, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor=color_f, alpha=0.75),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.2),
                    zorder=6
                )

            # Project raw point cloud as small dots
            if rock['points'] is not None:
                pts = rock['points']
                pts = pts[pts[:, 2] > 0.01]
                if len(pts):
                    us = (fx * pts[:, 0] / pts[:, 2] + cx).astype(int)
                    vs = (fy * pts[:, 1] / pts[:, 2] + cy).astype(int)
                    m  = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
                    ax.scatter(us[m], vs[m], s=1.5,
                               color=color_f, alpha=0.35, zorder=4)

        ax.set_title(
            f't = {frame.timestamp:.1f} s\n'
            f'{len(frame.rock_obs)} rock(s) detected',
            fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(out_dir, 'detection_examples.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: detection_examples.png")


# ─────────────────────────────────────────────────────────────
# Depth error computation
# ─────────────────────────────────────────────────────────────

def compute_depth_errors(frames: List[Frame]) -> List[dict]:
    """
    For each rock observation:
      stereo_depth = centroid.z  (from perception node)
      lidar_depth  = median LiDAR forward distance near centroid
      error        = |stereo_depth - lidar_depth|
    """
    results = []

    for frame in frames:
        if frame.camera_info is None:
            continue
        if frame.lidar_pts is None or len(frame.lidar_pts) == 0:
            continue

        for rock in frame.rock_obs:
            c            = rock['centroid']
            stereo_depth = float(c[2])

            if stereo_depth < 0.1 or stereo_depth > 5.0:
                continue

            lidar_depth = lidar_depth_for_rock(
                frame.lidar_pts, c,
                search_radius_m=0.8
            )
            if lidar_depth is None:
                continue

            error = abs(stereo_depth - lidar_depth)
            results.append({
                'timestamp':     round(frame.timestamp, 3),
                'stereo_depth':  round(stereo_depth, 4),
                'lidar_depth':   round(lidar_depth, 4),
                'abs_error':     round(error, 4),
                'rel_error_pct': round(100.0 * error / lidar_depth, 2),
            })

    return results


def save_depth_error_table(results: List[dict], out_dir: str):
    if not results:
        print("  WARNING: No depth error results to save")
        return

    errors    = np.array([r['abs_error']     for r in results])
    rel_errs  = np.array([r['rel_error_pct'] for r in results])
    depths    = np.array([r['lidar_depth']   for r in results])

    # ── Per-observation CSV ──────────────────────────────────
    obs_path = os.path.join(out_dir, 'depth_error.csv')
    with open(obs_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: depth_error.csv  ({len(results)} observations)")

    # ── Summary CSV ──────────────────────────────────────────
    summary = {
        'N observations':           len(results),
        'Mean abs error [m]':       round(float(np.mean(errors)),    4),
        'Median abs error [m]':     round(float(np.median(errors)),  4),
        'Std abs error [m]':        round(float(np.std(errors)),     4),
        'RMSE [m]':                 round(float(np.sqrt(np.mean(errors**2))), 4),
        'Max abs error [m]':        round(float(np.max(errors)),     4),
        'Min abs error [m]':        round(float(np.min(errors)),     4),
        'Mean rel error [%]':       round(float(np.mean(rel_errs)),  2),
        'Mean LiDAR depth [m]':     round(float(np.mean(depths)),    3),
        'Error < 0.10 m [%]':       round(float(np.mean(errors < 0.10) * 100), 1),
        'Error < 0.20 m [%]':       round(float(np.mean(errors < 0.20) * 100), 1),
        'Error < 0.50 m [%]':       round(float(np.mean(errors < 0.50) * 100), 1),
    }

    summary_path = os.path.join(out_dir, 'stereo_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Value'])
        for k, v in summary.items():
            w.writerow([k, v])

    print("\n" + "=" * 52)
    print("  STEREO DEPTH EVALUATION SUMMARY")
    print("=" * 52)
    for k, v in summary.items():
        print(f"  {k:<35} {v}")
    print("=" * 52)
    print(f"\n  Saved: stereo_summary.csv")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag',        required=True,
                        help='ROS2 bag directory')
    parser.add_argument('--model',      required=True,
                        help='Path to YOLO model (rock_seg.pt)')
    parser.add_argument('--out',        required=True,
                        help='Output directory')
    parser.add_argument('--n_examples', type=int, default=4,
                        help='Number of example frames per figure (default: 4)')
    parser.add_argument('--conf',       type=float, default=0.25,
                        help='YOLO confidence threshold (default: 0.25)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Bag:   {args.bag}")
    print(f"Model: {args.model}")
    print(f"Out:   {args.out}")

    # 1. Read bag
    print("\nReading bag...")
    frames = read_bag(args.bag)
    if not frames:
        print("ERROR: No frames found. Check bag topics.")
        sys.exit(1)

    # 2. Load YOLO
    print("\nLoading YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
        print("  Model loaded.")
    except Exception as e:
        print(f"  WARNING: Could not load YOLO model: {e}")
        model = None

    # 3. Depth map example
    print("\nGenerating depth map figure...")
    plot_depth_examples(frames, args.out)

    # 4. Detection + centroid overlay
    if model is not None:
        print("Generating detection overlay figure...")
        plot_detection_examples(frames, model,
                                1, args.out, args.conf)
    else:
        print("Skipping detection figure (no model).")

    # 5. Depth error table
    print("\nComputing stereo vs LiDAR depth errors...")
    results = compute_depth_errors(frames)
    save_depth_error_table(results, args.out)

    print(f"\nDone. All outputs in: {args.out}")


if __name__ == '__main__':
    main()