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

class MapExporter(Node):
    def __init__(self):
        super().__init__('map_exporter')
        
        self.map_resolution = 100  # 100 pixels per meter
        self.save_path = os.path.expanduser('/home/ws/ROS_ws/rock_map_cumulative.png')
        self.target_frame = 'odom' # The frame you want the map to be in
        
        # MEMORY: Key = Marker ID, Value = List of [x, y] in ODOM frame
        self.rock_memory = {} 
        
        # TF Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.sub = self.create_subscription(
            MarkerArray, 
            '/rock_markers', 
            self.marker_callback, 
            10
        )
        self.get_logger().info(f'TF-Aware Map Exporter Started. Target Frame: {self.target_frame}')

    def marker_callback(self, msg):
        if not msg.markers:
            return

        # 1. Get Transform (Camera -> Odom)
        # We assume all markers in the array are in the same frame (usually true)
        source_frame = msg.markers[0].header.frame_id
        try:
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rclpy.time.Time()
            )
        except Exception as e:
            # If TF fails (e.g. at startup), skip this frame
            return

        # 2. Process Markers
        for marker in msg.markers:
            if marker.type == Marker.POINTS:
                transformed_points = []
                
                for pt in marker.points:
                    # Create a PointStamped for TF
                    p_stamped = PointStamped()
                    p_stamped.point.x = pt.x
                    p_stamped.point.y = pt.y
                    p_stamped.point.z = pt.z
                    
                    # Transform! (This rotates Optical -> Odom automatically)
                    try:
                        p_out = do_transform_point(p_stamped, trans)
                        # We only keep X (Forward) and Y (Left) for the 2D map
                        transformed_points.append([p_out.point.x, p_out.point.y])
                    except:
                        pass
                
                # Update memory with correctly rotated points
                if transformed_points:
                    self.rock_memory[marker.id] = transformed_points

        # 3. Collect All Points
        all_points = []
        for pid, pts in self.rock_memory.items():
            all_points.extend(pts)

        if not all_points:
            return

        points_np = np.array(all_points)

        # 4. Auto-Scale
        min_x, min_y = np.min(points_np, axis=0)
        max_x, max_y = np.max(points_np, axis=0)
        
        min_x -= 2.0; max_x += 2.0
        min_y -= 2.0; max_y += 2.0

        width_m = max_x - min_x
        height_m = max_y - min_y
        
        img_w = int(width_m * self.map_resolution)
        img_h = int(height_m * self.map_resolution)
        
        # 5. Draw
        map_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        
        # Draw Origin (Red Cross)
        origin_u = int((0 - min_x) * self.map_resolution)
        origin_v = int((max_y - 0) * self.map_resolution) 
        if 0 <= origin_u < img_w and 0 <= origin_v < img_h:
            cv2.drawMarker(map_img, (origin_u, origin_v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Draw Rocks (Green)
        for pt in points_np:
            world_x, world_y = pt
            u = int((world_x - min_x) * self.map_resolution)
            v = int((max_y - world_y) * self.map_resolution) # Flip Y (Standard Image Coords)
            
            if 0 <= u < img_w and 0 <= v < img_h:
                cv2.circle(map_img, (u, v), 2, (0, 255, 0), -1)

        cv2.imwrite(self.save_path, map_img)
        self.get_logger().info(f'Map Saved. Rocks: {len(self.rock_memory)}', throttle_duration_sec=5.0)

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