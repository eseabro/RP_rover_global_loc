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
from visualization_msgs.msg import MarkerArray

class RockMatcherNode(Node):
    def __init__(self):
        super().__init__('rock_matcher_node')
        
        # --- Parameters ---
        self.declare_parameter('global_map_path', '/home/ws/src/hirise_data/marsyard2022_sat.csv')
        self.global_csv = self.get_parameter('global_map_path').get_parameter_value().string_value

        # Matcher Settings
        self.eps = 0.6
        self.binsize = 0.008
        self.size_tol = 0.35

        # --- Load Global Catalog ---
        self.load_global_catalog()

        # --- ROS Setup ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_data_sub = self.create_subscription(
            MarkerArray,
            '/ekf/landmarks',
            self.map_data_callback,
            10
        )
        
        self.match_cooldown = 5.0
        self.last_match_time_sec = -999.0
        # Publisher for EKF integration (The "Rock GPS")
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/rock_global_pose', 10)
        
        self.get_logger().info(f"Rock Matcher Node Started with eps={self.eps} and binsize={self.binsize}. Ready to localize.")
        

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
            from custom_slam.identifier import build_geometric_hash_fast
            self.get_logger().info("Pre-computing Geometric Hash Table...")
            self.global_hash = build_geometric_hash_fast(self.global_pts, self.global_sizes, binsize=self.binsize)
            # --------------------------------------------------------
            

            self.get_logger().info(f"Loaded {len(self.global_pts)} rocks from catalog.")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV: {e}")
            
    def map_data_callback(self, msg: MarkerArray):
        if self.catalog_dict is None or not msg.markers:
            return
        
        # 1. Extract the timestamp from the first marker's ROS header
        stamp = msg.markers[0].header.stamp
        self.last_landmark_stamp = stamp
        timestamp_sec = stamp.sec + (stamp.nanosec * 1e-9)

        # --- Time-Gated Logic & Sim Time Fallback ---
        if timestamp_sec == 0.0:
            if not hasattr(self, '_frame_fallback'): self._frame_fallback = 0
            self._frame_fallback += 1
            if self._frame_fallback % 50 != 0:
                return
        else:
            time_since_last = timestamp_sec - getattr(self, 'last_match_time_sec', 0.0)
            if time_since_last < self.match_cooldown: 
                return

        # 2. Unpack the EKF MarkerArray into Numpy lists
        local_pts_list = []
        local_sizes_list = []

        for marker in msg.markers:
            if marker.action == 3:  # Skip DELETEALL commands
                continue
            local_pts_list.append([marker.pose.position.x, marker.pose.position.y])
            
            local_sizes_list.append([marker.scale.x, marker.scale.y])
            

        local_pts = np.array(local_pts_list, dtype=np.float32)
        local_sizes = np.array(local_sizes_list, dtype=np.float32)
            
        if len(local_pts) < 5:
            return # Need at least a pentagon of rocks to match safely

        # 3. Prepare Input for Matcher
        sim_input = {
            'observed_vectors': local_pts,
            'observed_sizes': local_sizes,
            'n_false': int(len(local_pts) * 0.2) # Assume 80% match
        }
        min_inliers = max(5, int(len(local_pts) * 0.80))
        
        # 4. Run Geometric Matcher
        result = identify_geometric(
            sim_input, 
            self.catalog_dict,
            hash_index=self.global_hash,
            eps=self.eps,
            binsize=self.binsize,
            ransac_iters=4000,
            min_seed_inliers=min_inliers,
            early_exit_fraction=0.9,
            size_tolerance=self.size_tol
        )

        best = result.get('best_solution')
        iters = result.get('iterations', 'Unknown')
        early_exit = result.get('early_exit', False)
        self.last_match_time_sec = timestamp_sec
        # --- THE UPDATED AUTOPSY DEBUG BLOCK ---
        if best is None:
            self.get_logger().warn(f"💀 DEBUG: RANSAC failed entirely after {iters} iterations. Sizes or eps are likely too strict!")
            return

        inliers = best['inlier_count']
        s = best.get('s', 1.0)
        
        # Log exactly how hard RANSAC worked!
        exit_reason = "EARLY EXIT" if early_exit else "MAX ITERATIONS"
        self.get_logger().info(f"📊 DEBUG: RANSAC finished ({exit_reason} at {iters} iters) -> Best Inliers: {inliers}, Scale: {s:.2f}")

        # Gate 1: Did it find enough rocks?
        if inliers < min_inliers:
            self.get_logger().warn(f"❌ REJECTED: Not enough inliers ({inliers} < {min_inliers}). Saving failure plot.")
            self.save_debug_plot(self.catalog_dict['catalog_vectors'], self.catalog_dict['catalog_sizes'], 
                                 local_pts, local_sizes, best)
            return

        # Gate 2: Is the scale physically possible?
        if not (0.9 < s < 1.1):
            self.get_logger().warn(f"❌ REJECTED: Unrealistic Scale Factor (s={s:.2f}). Saving failure plot.")
            self.save_debug_plot(self.catalog_dict['catalog_vectors'], self.catalog_dict['catalog_sizes'], 
                                 local_pts, local_sizes, best)
            return
        
        # --- SUCCESS BLOCK ---
        self.publish_global_pose(best)
        self.get_logger().info(f"✅ SUCCESS! Propagating timestamp: {timestamp_sec:.2f}")
        
        self.save_debug_plot(self.catalog_dict['catalog_vectors'], self.catalog_dict['catalog_sizes'], 
                             local_pts, local_sizes, best)

            
    def publish_global_pose(self, best):
        """Calculates the rover's position in the global map frame and publishes it."""
        t = best['t']
        R = best['R']
        s = best.get('s', 1.0)
        inliers = best['inlier_count']
        
        try:
            # Get the rover's position in the SLAM 'map' frame
            trans = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time()
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

        # 1. Transform the rover's local 'map' position to the 'HiRISE global' position
        rover_local_vec = np.array([rover_x_local, rover_y_local])
        rover_global_pos = (s * (R @ rover_local_vec)) + t
        map_yaw = np.arctan2(R[1, 0], R[0, 0])
        rover_global_yaw = rover_yaw_local + map_yaw

        # --- THE FIX: THE DATUM ANCHOR ---
        # Save the global anchor point. Upgrade it if a much better match is found later.
        if not hasattr(self, 'datum_inliers') or inliers > getattr(self, 'datum_inliers', 0) + 2:
            self.datum_t = t
            self.datum_R = R
            self.datum_s = s
            self.datum_yaw = map_yaw
            self.datum_inliers = inliers
            self.get_logger().info(f"🛰️ Datum Set/Upgraded! Anchor at HiRISE X:{t[0]:.2f}, Y:{t[1]:.2f}")
            return # Let the datum settle for one frame
            
        # Convert the TRUE Global Pose back into the Local SLAM Frame
        R_datum_inv = np.linalg.inv(self.datum_R)
        local_gnss_pos = (1.0 / self.datum_s) * (R_datum_inv @ (rover_global_pos - self.datum_t))
        local_gnss_yaw = rover_global_yaw - self.datum_yaw
        # ---------------------------------
        
        self.get_logger().info(f"📍 EKF Local Pose: X={rover_x_local:.2f}, Y={rover_y_local:.2f}")
        self.get_logger().info(f"🌍 Drift Corrected Pose: X={local_gnss_pos[0]:.2f}, Y={local_gnss_pos[1]:.2f}")

        # Build the Pose Message
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = trans.header.stamp
        msg.header.frame_id = "map" 
        
        # Publish the local drift correction, NOT the raw HiRISE coordinate!
        msg.pose.pose.position.x = float(local_gnss_pos[0])
        msg.pose.pose.position.y = float(local_gnss_pos[1])
        msg.pose.pose.position.z = 0.0
        
        msg.pose.pose.orientation.z = float(np.sin(local_gnss_yaw / 2.0))
        msg.pose.pose.orientation.w = float(np.cos(local_gnss_yaw / 2.0))
        
        # Uncertainty based on inlier count
        uncertainty = 1.0 / (inliers + 1e-6)
        cov = np.zeros(36)
        cov[0] = uncertainty  
        cov[7] = uncertainty  
        cov[35] = uncertainty 
        msg.pose.covariance = cov.tolist()
        
        self.pose_pub.publish(msg)

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

            # Calculate the RANSAC map rotation in degrees
            map_yaw_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

            # 2. Draw Local Rocks 
            for i in range(len(transformed_local_pts)):
                x, y = transformed_local_pts[i]
                w, l = transformed_local_sizes[i]
                disp_w, disp_l = max(w, 0.05), max(l, 0.05)
                
                # --- THE FIX: Pass the angle so the shape rotates with the map! ---
                ax.add_patch(Ellipse((x, y), width=disp_w, height=disp_l, angle=map_yaw_deg,
                                     color='none', ec='orange', lw=2, linestyle='--'))
                # ------------------------------------------------------------------
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
            self.get_logger().debug(f"🖼️ Saved debug plot to {save_path}")
            
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