#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformListener
import cv2
import numpy as np
import os
import csv

class MapExporter(Node):
    def __init__(self):
        super().__init__('map_exporter')
        
        self.map_resolution = 100  # 100 pixels per meter
        self.image_save_path = os.path.expanduser('/home/ws/ROS_ws/rock_map_cumulative.png')
        self.csv_save_path = os.path.expanduser('/home/ws/ROS_ws/rock_analysis.csv')
        self.target_frame = 'odom' # Global Frame for the map
        
        # MEMORY: Key = Marker ID, Value = List of [x, y, z] in ODOM frame
        self.rock_memory = {}
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.sub = self.create_subscription(
            MarkerArray, 
            '/rock_markers', 
            self.marker_callback, 
            10
        )
        self.map_data_pub = self.create_publisher(Float32MultiArray, '/rock_map_data', 10)
        
        self.get_logger().info(f'Analysis Map Exporter Started.')
        self.get_logger().info(f'Image: {self.image_save_path}')
        self.get_logger().info(f'Data:  {self.csv_save_path}')


    def marker_callback(self, msg):
        if not msg.markers:
            return

        self.rock_memory.clear()

        # 1. Process Markers Directly
        for marker in msg.markers:

            if marker.action == 3:
                continue
                
            if marker.type == Marker.POINTS:
                raw_points = []
                
                for pt in marker.points:
                    raw_points.append([pt.x, pt.y, pt.z])
                
                if raw_points:
                    self.rock_memory[marker.id] = raw_points  
        
        # 2. Analyze & Export Data
        self.export_analysis()

        # 3. Draw Map (Standard Visualization)
        self.draw_map()
        
        # Publish Map Data
        self.publish_map_data()

    def publish_map_data(self):
        if not self.rock_memory:
            return
            
        msg = Float32MultiArray()
        flat_data = []
        
        for rock_id, points in self.rock_memory.items():
            pts_np = np.array(points)
            
            # Math from your export_analysis function
            min_bounds = np.min(pts_np, axis=0)
            max_bounds = np.max(pts_np, axis=0)
            dims = max_bounds - min_bounds
            center = np.mean(pts_np, axis=0)
            
            # Append the 7 values exactly as they would appear in the CSV
            flat_data.extend([
                float(rock_id),
                float(-center[1]), float(center[0]), float(center[2]), # X, Y, Z
                float(dims[0]), float(dims[1]), float(dims[2])        # W, L, H
            ])
            
        msg.data = flat_data
        self.map_data_pub.publish(msg)

    def export_analysis(self):
        """Calculates center and dimensions for every rock and saves to CSV"""
        if not self.rock_memory:
            return

        with open(self.csv_save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            writer.writerow(['ID', 'Map_X', 'Map_Y', 'Map_Z', 'Width_m', 'Length_m', 'Height_Z'])
            
            for rock_id, points in self.rock_memory.items():
                pts_np = np.array(points)
                
                # Calculate Bounds (Min/Max)
                min_bounds = np.min(pts_np, axis=0)
                max_bounds = np.max(pts_np, axis=0)
                
                # Dimensions (Size)
                dims = max_bounds - min_bounds
                width = dims[0]
                length = dims[1]
                height = dims[2]
                
                # Center Position
                center = np.mean(pts_np, axis=0)
                
                writer.writerow([
                    rock_id, 
                    f"{-center[1]:.2f}", f"{center[0]:.2f}", f"{center[2]:.2f}",
                    f"{width:.2f}", f"{length:.2f}", f"{height:.2f}"
                ])

    def draw_map(self):
        """Draws the top-down PNG map matching RViz orientation"""
        all_points = []
        
        # --- NEW: Add the (0,0) origin to the bounds so it never gets cropped! ---
        all_points.append([0.0, 0.0])
        # -------------------------------------------------------------------------
        
        for pid, pts in self.rock_memory.items():
            for p in pts:
                all_points.append([p[0], p[1]])

        if not all_points: return

        points_np = np.array(all_points)

        # Auto-Scale Bounds
        min_x, min_y = np.min(points_np, axis=0)
        max_x, max_y = np.max(points_np, axis=0)
        
        min_x -= 2.0; max_x += 2.0
        min_y -= 2.0; max_y += 2.0
        
        # --- FIX: Swap Width/Height to match ROS Coordinate conventions ---
        # Image width is controlled by ROS Y (Left/Right)
        width_m = max_y - min_y
        # Image height is controlled by ROS X (Forward/Backward)
        height_m = max_x - min_x
        # ------------------------------------------------------------------
        
        img_w = int(width_m * self.map_resolution)
        img_h = int(height_m * self.map_resolution)
        
        map_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        
        # --- FIX: RViz Image Mapping ---
        # OpenCV U (Left-to-Right) = Mapped from ROS +Y to -Y
        # OpenCV V (Top-to-Bottom) = Mapped from ROS +X to -X
        
        origin_u = int((max_y - 0.0) * self.map_resolution)
        origin_v = int((max_x - 0.0) * self.map_resolution) 
        
        if 0 <= origin_u < img_w and 0 <= origin_v < img_h:
            # Draws the red cross at the spawn origin
            cv2.drawMarker(map_img, (origin_u, origin_v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Draw Rocks
        for pt in points_np:
            u = int((max_y - pt[1]) * self.map_resolution)
            v = int((max_x - pt[0]) * self.map_resolution)
            
            if 0 <= u < img_w and 0 <= v < img_h:
                cv2.circle(map_img, (u, v), 2, (0, 255, 0), -1)

        cv2.imwrite(self.image_save_path, map_img)

def main(args=None):
    rclpy.init(args=args)
    node = MapExporter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()