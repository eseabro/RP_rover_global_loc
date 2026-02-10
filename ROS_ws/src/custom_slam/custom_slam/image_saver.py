import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
from datetime import datetime

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver_node')

        # --- Parameters ---
        # You can override these via command line if needed
        self.declare_parameter('save_folder', 'captured_images')
        self.declare_parameter('image_topic', '/camera/left/image_raw')
        self.declare_parameter('interval_seconds', 30.0)

        self.save_folder = self.get_parameter('save_folder').get_parameter_value().string_value
        self.topic_name = self.get_parameter('image_topic').get_parameter_value().string_value
        self.interval = self.get_parameter('interval_seconds').get_parameter_value().double_value

        # --- Setup ---
        # Create folder if it doesn't exist
        os.makedirs(self.save_folder, exist_ok=True)
        self.get_logger().info(f'Saving images to: {os.path.abspath(self.save_folder)}')

        # Tools
        self.bridge = CvBridge()
        self.latest_msg = None

        # Subscriber
        # We subscribe to the stream to keep 'latest_msg' fresh
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            10
        )

        # Timer
        # This triggers the actual saving logic every X seconds
        self.timer = self.create_timer(self.interval, self.timer_callback)
        self.get_logger().info(f'Subscribed to {self.topic_name}. Saving every {self.interval}s.')

    def image_callback(self, msg):
        """Just updates the buffer with the newest frame."""
        self.latest_msg = msg

    def timer_callback(self):
        """Periodically wakes up to save whatever is in the buffer."""
        if self.latest_msg is None:
            self.get_logger().warn('Timer fired, but no images received yet!')
            return

        try:
            # Convert ROS Image -> OpenCV Image
            # "bgr8" is standard for color images in OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_msg, desired_encoding='bgr8')

            # Generate a unique filename based on time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{timestamp}.jpg"
            filepath = os.path.join(self.save_folder, filename)

            # Save to disk
            cv2.imwrite(filepath, cv_image)
            self.get_logger().info(f'Saved: {filename}')

        except Exception as e:
            self.get_logger().error(f'Failed to save image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()