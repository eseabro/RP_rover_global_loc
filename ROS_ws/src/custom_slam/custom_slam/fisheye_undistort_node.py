#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer

class FisheyeUndistortNode(Node):
    def __init__(self):
        super().__init__('fisheye_undistort_node')
        self.bridge = CvBridge()

        # ══════════════════════════════════════════════════════════════════
        # 1. HARDCODED CALIBRATION FROM YOUR CALIB.YAML
        # ══════════════════════════════════════════════════════════════════
        # Image size
        self.W, self.H = 1280, 1024
        
        # Camera 0 (Left) Double Sphere Intrinsics
        # [xi, alpha, fx, fy, cx, cy]
        self.cam0_intrinsics = [-0.09835361, 0.60371944, 454.68404534, 454.15370656, 650.0322454, 532.01752155]
        
        # Camera 1 (Right) Double Sphere Intrinsics
        self.cam1_intrinsics = [-0.06728265, 0.61092767, 469.51305346, 469.09575048, 632.35494553, 506.29904867]

        # ══════════════════════════════════════════════════════════════════
        # 2. DEFINE THE NEW "PERFECT" PINHOLE CAMERA
        # ══════════════════════════════════════════════════════════════════
        # We are creating a virtual pinhole camera with a flat lens.
        self.new_fx = 600.0
        self.new_fy = 600.0
        self.new_cx = 640.0
        self.new_cy = 512.0
        
        self.get_logger().info("Pre-computing Double Sphere mapping grids. This takes a few seconds...")
        
        # Generate the highly-optimized mapping arrays
        self.map_x_left, self.map_y_left = self.create_ds_map(self.cam0_intrinsics)
        self.map_x_right, self.map_y_right = self.create_ds_map(self.cam1_intrinsics)
        
        self.get_logger().info("Mapping grids generated! Ready to rectify.")

        # ══════════════════════════════════════════════════════════════════
        # 3. ROS 2 PUBLISHERS & SUBSCRIBERS
        # ══════════════════════════════════════════════════════════════════
        self.left_pub = self.create_publisher(Image, '/camera/left/image_rect_color', 10)
        self.right_pub = self.create_publisher(Image, '/camera/right/image_rect_color', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/left/camera_info', 10)

        self.left_sub = Subscriber(self, Image, '/cam_0/image_raw')
        self.right_sub = Subscriber(self, Image, '/cam_1/image_raw')

        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], queue_size=10, slop=0.05
        )
        self.sync.registerCallback(self.sync_callback)


    def create_ds_map(self, intrinsics):
        """
        Calculates the reverse-projection from a perfect pinhole camera 
        back onto the distorted Double Sphere image plane using vectorized numpy.
        """
        xi, alpha, fx, fy, cx, cy = intrinsics
        
        # 1. Create a grid of every pixel in the new 1280x1024 image
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H))
        
        # 2. Convert to normalized pinhole coordinates
        x = (u - self.new_cx) / self.new_fx
        y = (v - self.new_cy) / self.new_fy
        z = np.ones_like(x)
        
        # 3. The Double Sphere Projection Math (Usenko et al. 2018)
        d1 = np.sqrt(x**2 + y**2 + z**2)
        d2 = np.sqrt(x**2 + y**2 + (z + xi * d1)**2)
        denominator = alpha * d2 + (1 - alpha) * (xi * d1 + z)
        
        # 4. Find where this perfect ray hits the actual distorted fisheye image
        map_x = fx * (x / denominator) + cx
        map_y = fy * (y / denominator) + cy
        
        return map_x.astype(np.float32), map_y.astype(np.float32)


    def sync_callback(self, left_msg, right_msg):
        try:
            # 1. Convert ROS messages to OpenCV format
            # Using mono8 because fisheye datasets are usually grayscale, 
            # we'll convert to bgr8 before publishing to keep YOLO happy!
            raw_left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
            raw_right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='mono8')

            # 2. Instant Un-distortion! (This takes < 2ms per image)
            rect_left = cv2.remap(raw_left, self.map_x_left, self.map_y_left, cv2.INTER_LINEAR)
            rect_right = cv2.remap(raw_right, self.map_x_right, self.map_y_right, cv2.INTER_LINEAR)
            
            # 3. Convert to BGR so your YOLO pipeline doesn't crash
            rect_left_color = cv2.cvtColor(rect_left, cv2.COLOR_GRAY2BGR)
            rect_right_color = cv2.cvtColor(rect_right, cv2.COLOR_GRAY2BGR)

            # 4. Repackage into ROS messages
            out_left = self.bridge.cv2_to_imgmsg(rect_left_color, encoding='bgr8')
            out_left.header = left_msg.header
            out_left.header.frame_id = 'cam_0_optical'

            out_right = self.bridge.cv2_to_imgmsg(rect_right_color, encoding='bgr8')
            out_right.header = right_msg.header
            out_right.header.frame_id = 'cam_1_optical'
            
            # 5. Build the CameraInfo message so your perception node knows the new focal length
            info_msg = CameraInfo()
            info_msg.header = out_left.header
            info_msg.width = self.W
            info_msg.height = self.H
            info_msg.distortion_model = "plumb_bob"
            info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0] # We already removed all distortion!
            info_msg.k = [
                self.new_fx, 0.0, self.new_cx,
                0.0, self.new_fy, self.new_cy,
                0.0, 0.0, 1.0
            ]
            info_msg.p = [
                self.new_fx, 0.0, self.new_cx, 0.0,
                0.0, self.new_fy, self.new_cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]

            # 6. Publish
            self.left_pub.publish(out_left)
            self.right_pub.publish(out_right)
            self.info_pub.publish(info_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to rectify images: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FisheyeUndistortNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()