import rclpy
from rclpy.node import Node
from rtabmap_msgs.msg import MapData, SensorData
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf_transformations
from image_geometry import PinholeCameraModel


class RGBDBEVProjector(Node):
    def __init__(self):
        super().__init__('rgbd_bev_projector')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(MapData, '/mapData', self.map_callback, 10)
        self.publisher = self.create_publisher(Image, '/rgbd_bev_map', 1)

        self.radius = 30.0  # meters
        self.resolution = 0.05  # meters per pixel
        self.size = int((2 * self.radius) / self.resolution)
        self.canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        self.get_logger().info("RGBD BEV Projector started")

    def map_callback(self, msg: MapData):
        self.canvas[:] = 0  # Clear canvas each callback
        for i, node in enumerate(msg.nodes):
            if i >= len(msg.rgb_images) or i >= len(msg.depth_images) or i >= len(msg.camera_infos):
                continue

            rgb_msg = msg.rgb_images[i]
            depth_msg = msg.depth_images[i]
            cam_info_msg = msg.camera_infos[i]
            pose = node.pose

            # Convert ROS images
            try:
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
                depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            except Exception as e:
                self.get_logger().warn(f"Failed to convert images: {e}")
                continue

            # Get 3D projection
            cam_model = PinholeCameraModel()
            cam_model.fromCameraInfo(cam_info_msg)

            try:
                Q = cam_model.projectionMatrix()
                points_3d = cv2.reprojectImageTo3D(depth, Q)
            except Exception as e:
                self.get_logger().warn(f"Reprojection failed: {e}")
                continue

            # Get pose
            tx = pose.position.x
            ty = pose.position.y
            tz = pose.position.z
            quat = pose.orientation
            rot = tf_transformations.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])[:3, :3]
            trans = np.array([tx, ty, tz])

            h, w = rgb.shape[:2]
            for v in range(0, h, 4):
                for u in range(0, w, 4):
                    x, y, z = points_3d[v, u]
                    if not np.isfinite(z) or z <= 0.2 or z > 5.0:
                        continue
                    pt = np.dot(rot, np.array([x, y, z])) + trans
                    ix = int((pt[0] + self.radius) / self.resolution)
                    iy = int((pt[1] + self.radius) / self.resolution)
                    if 0 <= ix < self.size and 0 <= iy < self.size:
                        self.canvas[iy, ix] = rgb[v, u]

        bev_msg = self.bridge.cv2_to_imgmsg(self.canvas, encoding='bgr8')
        self.publisher.publish(bev_msg)
        self.get_logger().info("Published BEV image")


def main(args=None):
    rclpy.init(args=args)
    node = RGBDBEVProjector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()