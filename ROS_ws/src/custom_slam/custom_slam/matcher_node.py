#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import Buffer, TransformListener
import numpy as np
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from custom_slam.identifier import identify_geometric

class RockMatcherNode(Node):
    def __init__(self):
        super().__init__('rock_matcher_node')
        
        # --- Parameters ---
        self.declare_parameter('global_map_path', '/home/ws/src/hirise_data/above_rock_analysis.csv')
        self.global_csv = self.get_parameter('global_map_path').get_parameter_value().string_value

        # Matcher Settings
        self.eps = 0.3
        self.binsize = 0.01

        # --- Load Global Catalog ---
        self.load_global_catalog()

        # --- ROS Setup ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to same topic as map_exporter
        self.map_data_sub = self.create_subscription(
            Float32MultiArray,
            '/rock_map_data',
            self.map_data_callback,
            10
        )
        
        self.match_cooldown = 5.0
        self.last_match_time_sec = -999.0
        # Publisher for EKF integration (The "Rock GPS")
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/rock_global_pose', 10)
        
        self.get_logger().info("Rock Matcher Node Started. Ready to localize.")
        

    def load_global_catalog(self):
        """Load the HiRISE satellite map using standard python csv module."""
        pts = []
        sizes = []
        
        try:
            with open(self.global_csv, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract coordinates
                    x = float(row['Map_X'])
                    y = float(row['Map_Y'])
                    w = float(row['Width_m'])
                    l = float(row['Length_m'])
                    
                    pts.append([x, y])
                    sizes.append([w, l])
                    
            self.global_pts = np.array(pts, dtype=np.float32)
            # Flip Y to match HiRISE orientation if needed
            self.global_pts[:, 1] = -self.global_pts[:, 1]
            
            self.global_sizes = np.array(sizes, dtype=np.float32)
            
            self.catalog_dict = {
                'catalog_vectors': self.global_pts,
                'catalog_sizes': self.global_sizes
            }
            self.get_logger().info(f"Loaded {len(self.global_pts)} rocks from catalog.")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV: {e}")
            


    def map_data_callback(self, msg):
        if self.catalog_dict is None:
            return
        
        # Ensure we have at least the timestamp + 1 rock (1 + 7 = 8 elements minimum)
        if not msg.data or len(msg.data) < 8:
            return 
        
        # 1. Extract the timestamp from the very front of the array!
        timestamp_sec = float(msg.data[0])
        
        # 2. Slice off the timestamp so the rest is pure rock data
        rock_data = msg.data[1:]

        # --- Time-Gated Logic & Sim Time Fallback ---
        if timestamp_sec == 0.0:
            if not hasattr(self, '_frame_fallback'): self._frame_fallback = 0
            self._frame_fallback += 1
            if self._frame_fallback % 50 != 0:
                return
        else:
            time_since_last = timestamp_sec - getattr(self, 'last_match_time_sec', 0.0)
            if time_since_last < getattr(self, 'match_cooldown', 1.0): # fallback to 1.0s if undefined
                return

        # 3. Reshape the rock data (now perfectly divisible by 7 again!)
        data_matrix = np.array(rock_data, dtype=np.float32).reshape(-1, 7)
        
        # Extract Local Map (X, Y) and Sizes (Width, Length)
        local_pts = data_matrix[:, 1:3]
        local_sizes = data_matrix[:, 4:6]
            
        if len(local_pts) < 5:
            return # Need at least a pentagon of rocks to match safely

        # 4. Prepare Input for Matcher
        sim_input = {
            'observed_vectors': local_pts,
            'observed_sizes': local_sizes,
            'n_false': int(len(local_pts) * 0.2) # Assume 80% match
        }
        min_inliers = len(local_pts) * 0.5
        
        # 5. Run Geometric Matcher
        result = identify_geometric(
            sim_input, 
            self.catalog_dict,
            eps=self.eps,
            binsize=self.binsize,
            ransac_iters=5000,
            min_seed_inliers=min_inliers,
            early_exit_fraction=0.7
        )

        best = result.get('best_solution')
        self.last_match_time_sec = timestamp_sec
        
        # 6. If Match Found, Publish Global Pose using the ORIGINAL timestamp
        if best and best['inlier_count'] >= min_inliers:
            
            self.publish_global_pose(best)
            self.get_logger().info(f"Found Match! Propagating timestamp: {timestamp_sec:.2f}")
            
            global_pts = self.catalog_dict['catalog_vectors']
            global_sizes = self.catalog_dict['catalog_sizes']
            
            # Run the plotting function in the background
            self.save_debug_plot(global_pts, global_sizes, local_pts, local_sizes, best)
            
        elif best:
            self.get_logger().info(f"No confident match found in this frame. Best: {best['inlier_count']}")
            
    def publish_global_pose(self, best):
        """Converts the RANSAC transform into the ROVER'S global pose."""
        t = best['t']
        R = best['R']
        
        # 1. Lookup the MOST RECENT rover position (bypass the future extrapolation error)
        try:
            # FIX: Use rclpy.time.Time() to get the latest available transform
            trans = self.tf_buffer.lookup_transform(
                'odom', 'base_footprint', rclpy.time.Time()
            )
            
            rover_x_local = trans.transform.translation.x
            rover_y_local = trans.transform.translation.y
            
            q = trans.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            rover_yaw_local = np.arctan2(siny_cosp, cosy_cosp)
            
        except Exception as e:
            self.get_logger().warn(f"Could not get rover's local pose: {e}")
            return

        # 2. Apply the RANSAC math to the rover's local position
        rover_local_vec = np.array([rover_x_local, rover_y_local])
        
        # X_global = R * X_local + t
        rover_global_pos = (R @ rover_local_vec) + t
        
        # The rover's global heading is its local heading plus the map rotation
        map_yaw = np.arctan2(R[1, 0], R[0, 0])
        rover_global_yaw = rover_yaw_local + map_yaw

        # 3. Build the Pose Message
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = trans.header.stamp
        msg.header.frame_id = "odom"
        
        # Set the TRUE Global Translation of the rover
        msg.pose.pose.position.x = float(rover_global_pos[0])
        msg.pose.pose.position.y = float(rover_global_pos[1])
        msg.pose.pose.position.z = 0.0
        
        # Set the TRUE Global Orientation of the rover
        msg.pose.pose.orientation.z = float(np.sin(rover_global_yaw / 2.0))
        msg.pose.pose.orientation.w = float(np.cos(rover_global_yaw / 2.0))
        
        # Covariance (unchanged)
        uncertainty = 1.0 / (best['inlier_count'] + 1e-6)
        cov = np.zeros(36)
        cov[0] = uncertainty  
        cov[7] = uncertainty  
        cov[35] = uncertainty 
        msg.pose.covariance = cov.tolist()
        
        self.pose_pub.publish(msg)
        self.get_logger().info(f"📍 Rover Global Pose: X={rover_global_pos[0]:.2f}, Y={rover_global_pos[1]:.2f} (Inliers: {best['inlier_count']})")


    def save_debug_plot(self, global_pts, global_sizes, local_pts, local_sizes, best_solution):
        """Transforms local rocks and saves an overlay image for debugging."""
        try:
            t = best_solution['t']
            R = best_solution['R']
            # Fallback to 1.0 if your RANSAC doesn't calculate scale
            s = best_solution.get('s', 1.0) 

            # Apply Similarity Transform: X_global = s * (R @ X_local) + t
            transformed_local_pts = (s * np.dot(R, local_pts.T)).T + t
            transformed_local_sizes = local_sizes * s

            fig, ax = plt.subplots(figsize=(10, 10))

            # 1. Draw Global Rocks (Solid gray ellipses)
            for i in range(len(global_pts)):
                x, y = global_pts[i]
                w, l = global_sizes[i]
                disp_w, disp_l = max(w, 0.05), max(l, 0.05)
                ax.add_patch(Ellipse((x, y), width=disp_w, height=disp_l, 
                                     color='lightgray', alpha=0.6, ec='gray'))

            # 2. Draw Local Rocks (Hollow orange dashed ellipses with an 'x')
            for i in range(len(transformed_local_pts)):
                x, y = transformed_local_pts[i]
                w, l = transformed_local_sizes[i]
                disp_w, disp_l = max(w, 0.05), max(l, 0.05)
                ax.add_patch(Ellipse((x, y), width=disp_w, height=disp_l, 
                                     color='none', ec='orange', lw=2, linestyle='--'))
                ax.scatter(x, y, c='orange', marker='x', s=30)

            # Formatting
            ax.set_title(f"RANSAC Match Overlay (Inliers: {best_solution['inlier_count']})")
            ax.set_xlabel("Global X (meters)")
            ax.set_ylabel("Global Y (meters)")
            ax.axis('equal') 
            ax.grid(True, alpha=0.3)

            # Save the image
            save_dir = os.path.expanduser('/home/ws/ROS_ws/')

            filename = f"match_overlay.png"
            save_path = os.path.join(save_dir, filename)
            
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
            # CRITICAL: Close the figure to free up RAM!
            plt.close(fig)
            self.get_logger().info(f"🖼️ Saved debug plot to {save_path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to generate debug plot: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = RockMatcherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()