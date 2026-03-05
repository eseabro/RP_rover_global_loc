#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

class OdomToPath(Node):
    def __init__(self):
        super().__init__('odom_to_path_node')
        
        # Listen to Gazebo's Ground Truth Odometry
        self.sub = self.create_subscription(
            Odometry, 
            '/ground_truth/odom', 
            self.odom_callback, 
            10
        )
        
        # Publish the RViz-friendly Path
        self.pub = self.create_publisher(Path, '/ground_truth/path', 10)
        
        self.path_msg = Path()

    def odom_callback(self, msg: Odometry):
        # 1. Sync the headers
        self.path_msg.header = msg.header
        
        # 2. Extract the Pose and wrap it in a PoseStamped
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        
        # 3. Append to the Path array
        self.path_msg.poses.append(pose_stamped)
        
        # 4. Memory management (Keep last 5000 points)
        if len(self.path_msg.poses) > 5000:
            self.path_msg.poses.pop(0)
            
        # 5. Publish!
        self.pub.publish(self.path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OdomToPath()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()