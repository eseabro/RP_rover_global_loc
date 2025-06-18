import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import numpy as np
import cv2
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import struct
import os 


class CloudMapBEVNode(Node):
    def __init__(self):
        super().__init__('cloud_map_bev_node')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/cloud_map',
            self.cloud_map_callback,
            10)

        self.bev_pub = self.create_publisher(Image, '/cloud_map_bev', 10)

        self.bridge = CvBridge()

        # BEV parameters
        self.radius = 30.0  # meters around robot to include
        self.resolution = 0.1  # meters per pixel

        # Image size (pixels)
        self.img_size = int((2 * self.radius) / self.resolution)

        self.get_logger().info("CloudMapBEVNode started")

    def unpack_rgb(self, rgb_float):
        # Converts packed float RGB into tuple (R,G,B)
        s = struct.pack('>f', rgb_float)
        i = struct.unpack('>I', s)[0]
        r = (i >> 16) & 0x0000ff
        g = (i >> 8) & 0x0000ff
        b = (i) & 0x0000ff
        return (r, g, b)

    def save_image_on_shutdown(self):
        if self.latest_image is not None:
            output_path = os.path.expanduser('./src/custom_slam/images/last_bev_image.png')
            cv2.imwrite(output_path, self.latest_image)
            self.get_logger().info(f"Saved latest image to {output_path}")
        else:
            self.get_logger().warn("No image was received; nothing to save.")

    def cloud_map_callback(self, msg):
        # Create blank image (3-channel BGR)
        bev_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Read points from PointCloud2
        points = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "rgb"))

        for p in points:
            x, y, z, rgb_float = p

            # Filter points within radius
            if abs(x) > self.radius or abs(y) > self.radius:
                continue

            # Convert 3D XY to pixel indices
            ix = int((x + self.radius) / self.resolution)
            iy = int((y + self.radius) / self.resolution)

            # Convert float RGB to tuple
            r, g, b = self.unpack_rgb(rgb_float)

            # Paint pixel (OpenCV uses BGR)
            bev_img[self.img_size - iy - 1, ix] = (b, g, r)

        # Convert to ROS Image and publish
        self.latest_image = bev_img.copy()
        img_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding="bgr8")
        img_msg.header = msg.header
        self.bev_pub.publish(img_msg)
        self.get_logger().info("Published BEV image")

def main(args=None):
    rclpy.init(args=args)
    node = CloudMapBEVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_image_on_shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
