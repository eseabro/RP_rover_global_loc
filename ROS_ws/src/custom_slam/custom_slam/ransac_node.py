import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from cv_bridge import CvBridge
import cv2

from .ransac_matcher import RANSACMatcher

class RANSAC_Node(Node):
    def __init__(self):
        super().__init__('ransac_node')
        self.bridge = CvBridge()
        self.declare_parameter('ransac_threshold', 0.1)
        self.declare_parameter('ransac_max_trials', 1000)
        self.ransac = RANSACMatcher(threshold=0.1, max_trials=1000)

        map_path = self.declare_parameter('global_map_path', 'src/img/global_map.png').get_parameter_value().string_value
        self.global_map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

        if self.global_map_img is None:
            self.get_logger().error(f"Failed to load global map from {map_path}")
        else:
            self.get_logger().info(f"Loaded global map from {map_path}")

        # Extract keypoints or features (you'll adapt this to what your RANSACMatcher needs)
        self.global_map_points = self.extract_features_from_image(self.global_map_img)

        self.bev_pub = self.create_subscription(Image, '/local_bev_map', self.map_callback, 10)
        # self.map_sub = self.create_subscription(Image, '/global_map', self.global_map_callback, 10)

        # self.bev_pub = self.create_publisher(Image, '/local_bev_map', 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/localization_pose', self.pose_callback, 10)
        self.match = self.create_publisher(Image, '/ransac_match', 10)
        self.pose = None
        self.map_points = []

        self.get_logger().info("RANSAC Matcher Node started")

    def pose_callback(self, msg):
        self.pose = msg
        self.get_logger().info(f"Pose received: {self.pose.pose.pose.position.x}, {self.pose.pose.pose.position.y}, {self.pose.pose.pose.position.z}")

    def global_map_callback(self, msg):
        self.global_map = msg
        self.get_logger().info("Global map received")

    def map_callback(self, msg):
        self.map_points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(msg, skip_nans=True)])
        if self.pose is None:
            return
        self.get_logger().info("Map received")
        # Extract pose translation
        pos = self.pose.pose.pose.position
        t = np.array([pos.x, pos.y, pos.z])
        # Filter points within radius
        radius = 30.0
        local_pts = np.array([p for p in self.map_points if np.linalg.norm(p - t) < radius])
        global_pts = self.global_map_points  # ✅ New: using preloaded map

        # Perform RANSAC matching
        inliers, outliers = self.ransac.match(global_pts, local_pts)

        self.get_logger().info(f"RANSAC inliers: {np.sum(inliers)}, outliers: {np.sum(outliers)}")
        # Publish RANSAC match
        ransac_img = np.zeros((480, 640), dtype=np.uint8)
        ransac_img[inliers] = 255
        ransac_img_msg = self.bridge.cv2_to_imgmsg(ransac_img, encoding='mono8')
        self.match.publish(ransac_img_msg)
        self.get_logger().info("RANSAC match published")

    def set_threshold(self, threshold):
        self.ransac.set_threshold(threshold)
        self.get_logger().info(f"RANSAC threshold set to: {self.ransac.threshold}")

    def set_max_trials(self, max_trials):
        self.ransac.max_trials = max_trials
        self.get_logger().info(f"RANSAC max trials set to: {self.ransac.max_trials}")

def main(args=None):
    rclpy.init(args=args)
    node = RANSAC_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
