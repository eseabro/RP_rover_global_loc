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
from nav_msgs.msg import Odometry
import math
import time
import os
import csv

class RockMatcherNode(Node):
    def __init__(self):
        super().__init__('rock_matcher_node')
        
        
        # --- Parameters ---
        MY = 'CNES'
        self.declare_parameter('global_map_path', f'/home/ws/src/hirise_data/marsyard{MY}_sat_adjusted.csv')
        self.global_csv = self.get_parameter('global_map_path').get_parameter_value().string_value

        # Matcher Settings
        self.eps = 0.15 # 0.6 for 2021, 0.9 for 2022
        self.binsize = 0.01 # before 0.01
        self.size_tol = 0.5
        
        # --- Load Global Catalog ---
        self.load_global_catalog()

        # --- ROS Setup ---
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_data_sub = self.create_subscription(
            MarkerArray,
            '/ekf/landmarks',
            self.map_data_callback,
            10
        )

        self.match_cooldown = 3.0
        self.last_match_time_sec = -999.0
        
        # Publisher for EKF integration (The "Rock GPS") x: -19.38592887840357 y: -12.515103795568585
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/rock_global_pose', 10)
        
        self.get_logger().info(f"Rock Matcher Node Started with eps={self.eps} and binsize={self.binsize}. Ready to localize.")
        
        # --- PERFORMANCE LOGGING SETUP ---
        self.metrics_file_path = os.path.expanduser(f'/home/ws/src/matcher_csvs/MY{MY}.csv')
        file_exists = os.path.exists(self.metrics_file_path)
        
        self.metrics_file = open(self.metrics_file_path, 'a+', newline='')
        self.csv_writer = csv.writer(self.metrics_file)
        
        if not file_exists:
            # Write the header row
            self.csv_writer.writerow([
                'timestamp_sec', 'compute_time_ms', 'local_map_size', 
                'dynamic_eps', 'iterations_used', 'early_exit_triggered', 
                'inlier_count', 'rmse', 'status'
            ])

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
            
            self.global_pts[:, 1] = -self.global_pts[:, 1]
            self.global_pts[:, 0] = -self.global_pts[:, 0]
                
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
        
        stamp = msg.markers[0].header.stamp
        timestamp_sec = stamp.sec + (stamp.nanosec * 1e-9)

        # --- Time-Gated Logic ---
        if timestamp_sec == 0.0:
            if not hasattr(self, '_frame_fallback'): self._frame_fallback = 0
            self._frame_fallback += 1
            if self._frame_fallback % 50 != 0: return
        else:
            time_since_last = timestamp_sec - getattr(self, 'last_match_time_sec', 0.0)
            if time_since_last < self.match_cooldown: 
                return

        try:
            trans = self.tf_buffer.lookup_transform('map', 'ekf_base_footprint', stamp)
            rover_vec = np.array([trans.transform.translation.x, trans.transform.translation.y])
            
        except Exception as e:
            self.get_logger().debug(f"TF Traffic Jam! Skipping frame to catch up. ({e})")
            return
        
        # ------------------------------------

        # Unpack the EKF MarkerArray into Numpy lists
        local_pts_list = []
        local_sizes_list = []

        for marker in msg.markers:
            if marker.action == 3: continue

            local_pts_list.append([marker.pose.position.x, marker.pose.position.y])
            local_sizes_list.append([marker.scale.x, marker.scale.y])

        local_pts = np.array(local_pts_list, dtype=np.float32)
        local_sizes = np.array(local_sizes_list, dtype=np.float32)
        if len(local_pts) < 5: return

        if len(local_pts) > 10:
            confidence_radius = 12.0 
            min_rocks = 3
            max_rocks = 50 
            
            # 1. Calculate physical 2D distance of every rock to the rover
            distances = np.linalg.norm(local_pts - rover_vec, axis=1)

            # 2. Sort the indices from closest (0.0m) to furthest
            sorted_idxs = np.argsort(distances)

            # 3. Count how many rocks are naturally inside our safe stereo radius
            rocks_in_radius = int(np.sum(distances <= confidence_radius))

            # 4. Take the rocks in the radius, OR the minimum required amount
            num_to_keep = max(rocks_in_radius, min_rocks)

            # 5. Cap it at max_rocks just in case we hit a massive dense cluster
            num_to_keep = min(num_to_keep, max_rocks)

            # 6. Slice the sorted array. If num_to_keep > rocks_in_radius, 
            #    it naturally "spills over" and grabs the closest outside rocks!
            keep_idxs = sorted_idxs[:num_to_keep]

            local_pts = local_pts[keep_idxs]
            local_sizes = local_sizes[keep_idxs]
        # ══════════════════════════════════════════════════════════════


        # 3. Prepare Input for Matcher
        sim_input = {
            'observed_vectors': local_pts,     
            'observed_sizes': local_sizes,     
            'n_false': int(len(local_pts) * 0.2) 
        }
               
        
        map_spread_x = np.max(local_pts[:, 0]) - np.min(local_pts[:, 0])
        map_spread_y = np.max(local_pts[:, 1]) - np.min(local_pts[:, 1])
        map_size_meters = max(map_spread_x, map_spread_y)
        
        # 2. Scale EPS: Base is 0.8m. Add 3cm of tolerance for every 1 meter the map grows.
        dynamic_eps = min(self.eps + (map_size_meters * 0.005), 1.0) 
        self.get_logger().debug(f"📏 Map Spread: {map_size_meters:.1f}m | Dynamic EPS: {dynamic_eps:.2f}")
        
        
        min_inliers = max(6, min(int(len(local_pts) * 0.1), 50))
        local_size = len(local_pts)
        
        ransac_start = time.perf_counter()

        result = identify_geometric(
            sim_input, 
            self.catalog_dict,
            hash_index=self.global_hash,
            eps=dynamic_eps,
            binsize=self.binsize,
            ransac_iters=2000,
            inv_neighbor_radius=0,
            min_seed_inliers=min_inliers,
            early_exit_fraction=0.85,
            size_tolerance=self.size_tol,
            prior_pos=None
        )
        
        ransac_end = time.perf_counter()
        compute_time_ms = (ransac_end - ransac_start) * 1000.0

        best = result.get('best_solution')
        iters = result.get('iterations', 'MAX')
        early_exit = result.get('early_exit', False)
        self.last_match_time_sec = timestamp_sec

        if best is None:
            self.get_logger().warn(f"💀 DEBUG: RANSAC failed entirely after {iters} iterations. Sizes or eps are likely too strict!")
            return

        inliers = best['inlier_count']
        s = best.get('s', 1.0)
        rms = best.get('rms', 0.0)
        
        exit_reason = "EARLY EXIT" if early_exit else "MAX ITERATIONS"
        self.get_logger().debug(f"📊 DEBUG: RANSAC finished ({exit_reason} at {iters} iters) -> Best Inliers: {inliers}/{len(local_pts)}, Scale: {s:.2f}, RMSE: {rms:.3f}m ")

        # =========================================================
        # 3. LOG TO CSV
        # =========================================================
        self.csv_writer.writerow([
            timestamp_sec, 
            round(compute_time_ms, 3), 
            local_size, 
            round(dynamic_eps, 3), 
            iters, 
            early_exit, 
            inliers, 
            round(rms, 4), 
            'SUCCESS'
        ])
        self.metrics_file.flush() 

        self.publish_global_pose(best, trans, stamp)
        self.save_debug_plot(self.catalog_dict['catalog_vectors'], self.catalog_dict['catalog_sizes'], 
                             local_pts, local_sizes, best, rover_vec)

            
    def publish_global_pose(self, best, rover_trans, stamp):
        t = best['t']
        R = best['R']
        s = best.get('s', 1.0)
        inliers = best['inlier_count']
        rms = best['rms']
        
        try:
            if not rover_trans:
                rover_trans = self.tf_buffer.lookup_transform('map', 'ekf_base_footprint', stamp)
                self.get_logger().info("GEtting new transform - breaking pipeline")
            rover_vec = np.array([rover_trans.transform.translation.x, rover_trans.transform.translation.y])
            
            q = rover_trans.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            rover_yaw_local = np.arctan2(siny_cosp, cosy_cosp)
            
        except Exception as e:
            self.get_logger().warn(f"Could not get rover's local pose: {e}")
            return

        # 2. Pure matrix math (Calculates the exact coordinate in the CSV frame)
        pure_global_pos = (s * np.dot(R, rover_vec)) + t
                
        rover_global_x = -float(pure_global_pos[0])
        rover_global_y = float(pure_global_pos[1])

        # F = np.array([[0, 1], [1, 0]])
        # R_rot = F @ R                         # was R @ F
        # map_yaw = np.arctan2(R_rot[1, 0], R_rot[0, 0])
        # pure_global_yaw = rover_yaw_local + map_yaw
        # rover_global_yaw = (pure_global_yaw + np.pi) % (2*np.pi) - np.pi/2
        # rover_global_yaw = pure_global_yaw
        
        # 1. Create a unit vector pointing in the rover's current local heading
        v_local = np.array([math.cos(rover_yaw_local), math.sin(rover_yaw_local)])
        
        # 2. Rotate the vector using the exact same R matrix from your matcher
        v_raw = R @ v_local
        
        # 3. Apply the exact same X-axis reflection you applied to your position
        rover_global_dir_x = -float(v_raw[0])
        rover_global_dir_y = float(v_raw[1])
        
        # 4. Read the final angle directly from the transformed vector
        rover_global_yaw = math.atan2(rover_global_dir_y, rover_global_dir_x)

        # ══════════════════════════════════════════════════════════════
        # THE MATCHER GATE
        # ══════════════════════════════════════════════════════════════
        # 1. Compare the newly calculated global pose to the EKF's current TF pose
        dx = rover_global_x - rover_vec[0]
        dy = rover_global_y - rover_vec[1]
        phys_dist = math.hypot(dx, dy)
        
        # 2. Calculate the Yaw difference and normalize it between -pi and pi
        raw_dyaw = rover_global_yaw - rover_yaw_local
        dyaw = math.atan2(math.sin(raw_dyaw), math.cos(raw_dyaw))


        # We are localized! Block any RANSAC hallucinations from being published.
        if phys_dist > 7.0: 
            self.get_logger().warn(f"🚫 Matcher Blocked Publish! Jump too large: {phys_dist:.1f}m > 5.0m")
            # self.get_logger().warn(f"Current Rover EKF: x:{rover_vec[0]} y:{rover_vec[1]}")
            # self.get_logger().warn(f"Mine: {rover_global_x}, {rover_global_y}")
            return

        if abs(dyaw) > math.radians(30.0):
            self.get_logger().warn(f"🚫 Matcher Blocked Publish! Angle jump too large: {math.degrees(dyaw):.1f}°")
            return

        # # If the average error per rock is greater than 40cm, it's likely a hallucination
        # if rms > 0.40:
        #     self.get_logger().warn(f"🚫 Matcher Blocked! RMSE too high: {rms:.2f}m > 0.40m")
        #     return
        
        # Build the Pose Message
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        
        # Publish the raw HiRISE absolute coordinates
        msg.pose.pose.position.x = float(rover_global_x)
        msg.pose.pose.position.y = float(rover_global_y)
        msg.pose.pose.position.z = 0.0
        
        msg.pose.pose.orientation.z = float(np.sin(rover_global_yaw / 2.0))
        msg.pose.pose.orientation.w = float(np.cos(rover_global_yaw / 2.0))
        
        # Uncertainty based on inlier count
        uncertainty = min(max(rms**2, 0.005), (1.0 / (inliers + 1e-6)))
        cov = np.zeros(36)
        cov[0] = uncertainty  
        cov[7] = uncertainty  
        cov[35] = uncertainty 
        msg.pose.covariance = cov.tolist()
        
        self.pose_pub.publish(msg)
        self.get_logger().debug(f"📍 EKF Local Pose: X={rover_vec[0]:.2f}, Y={rover_vec[1]:.2f}")
        self.get_logger().debug(f"🌍 Absolute Global Pose: X={rover_global_x:.2f}, Y={rover_global_y:.2f}")
        self.get_logger().info(f"🌍 Absolute Global Reloc: X={rover_vec[0]:.2f}, Y={rover_vec[1]:.2f}, RMS: {rms}, n inliers: {inliers}")

    def save_debug_plot(self, global_pts, global_sizes, local_pts, local_sizes, best_solution, rover_local_pos=None):
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
            ax.invert_xaxis()

            # Calculate the RANSAC map rotation in degrees
            map_yaw_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

            # 2. Draw Local Rocks 
            for i in range(len(transformed_local_pts)):
                x, y = transformed_local_pts[i]
                w, l = transformed_local_sizes[i]
                
                disp_w, disp_l = max(w, 0.05), max(l, 0.05)
                
                ax.add_patch(Ellipse((x, y), width=disp_w, height=disp_l, angle=map_yaw_deg,
                                     fill=False, edgecolor='orange', lw=2, linestyle='--'))
                
                ax.scatter(x, y, c='orange', marker='x', s=5)


            matches = best_solution.get('matches', [])
            for match in matches:
                g_idx, l_idx = int(match[0]), int(match[1])
                
                gx, gy = global_pts[g_idx]
                lx, ly = transformed_local_pts[l_idx]
                
                ax.plot([gx, lx], [gy, ly], color='lime', linewidth=1)
                ax.scatter(gx, gy, c='lime', marker='s', s=20, edgecolors='black')
                

            # 1. Draw Global Rocks (Solid gray ellipses)
            for i in range(len(global_pts)):
                x, y = global_pts[i]
                w, l = global_sizes[i]
                disp_w, disp_l = max(w, 0.05), max(l, 0.05)
                ax.add_patch(Ellipse((x, y), width=disp_w, height=disp_l, 
                                     color='lightgray', alpha=0.6, ec='gray'))
            # ══════════════════════════════════════════════════════════════
            # 3. Draw the Rover (The Blue Circle)
            # ══════════════════════════════════════════════════════════════
            if rover_local_pos is not None:
                # Transform the rover's local coordinate into the global satellite frame
                rover_global = (s * np.dot(R, rover_local_pos)) + t
                
                self.get_logger().info(f"Rover MAP position is {rover_global[0]}, {rover_global[1]}")
                
                # Plot a distinct blue circle with a white border
                ax.scatter(rover_global[0], rover_global[1], 
                           c='blue', marker='o', s=50, edgecolors='white', 
                           linewidth=2, zorder=5, label='Rover')
                ax.legend(loc='upper right')
            # ══════════════════════════════════════════════════════════════

            # Formatting
            ax.set_title(f"RANSAC Match Overlay (Inliers: {best_solution['inlier_count']})")
            ax.set_xlabel("Global X (meters)")
            ax.set_ylabel("Global Y (meters)")
            ax.axis('equal') 
            ax.grid(True, alpha=0.3)

            # Save the image
            import os
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