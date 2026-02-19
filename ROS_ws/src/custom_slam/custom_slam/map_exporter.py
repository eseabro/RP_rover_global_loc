#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
import cv2
import numpy as np
import os
import csv
import math

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
        self.get_logger().info(f'Analysis Map Exporter Started.')
        self.get_logger().info(f'Image: {self.image_save_path}')
        self.get_logger().info(f'Data:  {self.csv_save_path}')

    def marker_callback(self, msg):
        if not msg.markers:
            return

        # 1. Get Transform (Camera -> Odom)
        first_marker = msg.markers[0]
        source_frame = first_marker.header.frame_id
        exact_time = first_marker.header.stamp
        
        try:
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                exact_time
            )
        except Exception:
            return

        # 2. Process Markers (Accumulate Points)
        for marker in msg.markers:
            if marker.type == Marker.POINTS:
                transformed_points = []
                
                for pt in marker.points:
                    p_stamped = PointStamped()
                    p_stamped.point.x = pt.x
                    p_stamped.point.y = pt.y
                    p_stamped.point.z = pt.z
                    
                    try:
                        p_out = do_transform_point(p_stamped, trans)
                        transformed_points.append([p_out.point.x, p_out.point.y, p_out.point.z])
                    except:
                        pass
                
                if transformed_points:
                    self.rock_memory[marker.id] = transformed_points

        # 3. Analyze & Export Data
        self.export_analysis()

        # 4. Draw Map (Standard Visualization)
        self.draw_map()

    def export_analysis(self):
        """Calculates center and dimensions for every rock and saves to CSV"""
        if not self.rock_memory:
            return

        with open(self.csv_save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            writer.writerow(['ID', 'Map_X', 'Map_Y', 'Map_Z', 'Width_X', 'Length_Y', 'Height_Z', 'Distance_From_Start'])
            
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
                    f"{center[0]:.2f}", f"{center[1]:.2f}", f"{center[2]:.2f}",
                    f"{width:.2f}", f"{length:.2f}", f"{height:.2f}"
                ])

    def draw_map(self):
        """Draws the top-down PNG map"""
        all_points = []
        for pid, pts in self.rock_memory.items():
            # We only need X, Y for the map image
            for p in pts:
                all_points.append([p[0], p[1]])

        if not all_points: return

        points_np = np.array(all_points)

        # Auto-Scale
        min_x, min_y = np.min(points_np, axis=0)
        max_x, max_y = np.max(points_np, axis=0)
        
        min_x -= 2.0; max_x += 2.0
        min_y -= 2.0; max_y += 2.0
        
        width_m = max_x - min_x
        height_m = max_y - min_y
        
        img_w = int(width_m * self.map_resolution)
        img_h = int(height_m * self.map_resolution)
        
        map_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        
        # Origin
        origin_u = int((0 - min_x) * self.map_resolution)
        origin_v = int((max_y - 0) * self.map_resolution) 
        if 0 <= origin_u < img_w and 0 <= origin_v < img_h:
            cv2.drawMarker(map_img, (origin_u, origin_v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Draw Rocks
        for pt in points_np:
            u = int((pt[0] - min_x) * self.map_resolution)
            v = int((max_y - pt[1]) * self.map_resolution)
            
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