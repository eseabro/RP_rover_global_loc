import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from cv_bridge import CvBridge

from .bev_builder import GlobalBEVBuilder, build_local_bev_map, extract_bev_features

class BEVMapperNode(Node):
    def __init__(self):
        super().__init__('bev_mapper')
        self.bridge = CvBridge()

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/localization_pose', self.pose_callback, 10)
        self.map_sub = self.create_subscription(PointCloud2, '/cloud_map', self.map_callback, 10)

        self.bev_pub = self.create_publisher(Image, '/local_bev_map', 10)
        self.pose = None
        self.map_points = []

    def pose_callback(self, msg):
        self.pose = msg
        self.get_logger().info(f"Pose received: {self.pose.pose.pose.position.x}, {self.pose.pose.pose.position.y}, {self.pose.pose.pose.position.z}")

    def map_callback(self, msg):
        if self.pose is None:
            return
        self.get_logger().info("Map received")

        import sensor_msgs_py.point_cloud2 as pc2
        self.map_points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(msg, skip_nans=True)])

        # Extract pose translation
        pos = self.pose.pose.pose.position
        t = np.array([pos.x, pos.y, pos.z])

        # Filter points within radius
        radius = 30.0
        local_pts = np.array([p for p in self.map_points if np.linalg.norm(p - t) < radius])

        # Build BEV
        local_bev_map = build_local_bev_map(local_pts)

        # Publish BEV image
        bev_img_msg = self.bridge.cv2_to_imgmsg(local_bev_map, encoding='mono8')
        self.bev_pub.publish(bev_img_msg)
        self.get_logger().info("BEV map published")


def main(args=None):
    rclpy.init(args=args)
    node = BEVMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
