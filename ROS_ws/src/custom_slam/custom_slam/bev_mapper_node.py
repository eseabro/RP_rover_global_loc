import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import math

from .bev_builder import build_local_bev_map, build_colour_bev_map

class BEVMapperNode(Node):
    def __init__(self):
        super().__init__('bev_mapper')
        self.bridge = CvBridge()

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/localization_pose', self.pose_callback, 10)
        self.map_sub = self.create_subscription(PointCloud2, '/cloud_map', self.map_callback, 10)

        self.bev_pub = self.create_publisher(Image, '/local_bev_map', 10)
        self.get_logger().info("BEV Mapper Node started")
        self.pose = None
        self.map_points = []

    def pose_callback(self, msg):
        self.pose = msg
        self.get_logger().info(f"Pose received: {self.pose.pose.pose.position.x}, {self.pose.pose.pose.position.y}, {self.pose.pose.pose.position.z}")

    def map_callback(self, msg):
        if self.pose is None:
            return
        self.get_logger().info("Map received")

        points = []
        for p in pc2.read_points(msg, field_names=["x", "y", "z", "rgb"], skip_nans=True):
            
            x, y, z, rgb = p
            if math.isnan(rgb):  # Skip if RGB is NaN
                continue
            rgb = int(rgb)
            r = (rgb >> 16) & 255
            g = (rgb >> 8) & 255
            b = rgb & 255
            points.append([x, y, z, r, g, b])
        self.map_points = np.array(points)
        self.get_logger().info(f"Map points extracted: {len(self.map_points)}")

        # Extract pose translation
        pos = self.pose.pose.pose.position
        t = np.array([pos.x, pos.y, pos.z])

        # Filter points within radius
        radius = 30.0
        local_pts = np.array([p for p in self.map_points if np.linalg.norm(p - t) < radius])

        self.get_logger().info("Sample RGB values from local_pts:")
        self.get_logger().info(local_pts[:10, 3:6])  # Expecting something like [[123 45 200] ...]

        # Build BEV
        local_bev_map = build_colour_bev_map(local_pts)

        # Publish BEV image
        bev_img_msg = self.bridge.cv2_to_imgmsg(local_bev_map, encoding='bgr8')
        # print("BEV shape:", local_bev_map.shape)
        # print("dtype:", local_bev_map.dtype)

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
