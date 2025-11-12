import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Quaternion
from tf_transformations import quaternion_from_matrix
import numpy as np
from pyramid_identifier.identifier import identify
from pyramid_identifier.catalog import build_catalog
from pyramid_identifier.pyramids import build_kvector

class IdentifierNode(Node):
    def __init__(self):
        super().__init__('identifier')
        self.catalog = build_catalog(n=60, seed=1)
        self.catalog_index = build_kvector(self.catalog)
        self.subscription = self.create_subscription(
            Float32MultiArray,  # or your custom message
            '/observed_vectors',
            self.callback,
            10)
        self.publisher = self.create_publisher(Quaternion, '/rotation_estimate', 10)


    def callback(self, msg):
        print("Received observed vectors")
        obs = np.array(msg.data).reshape(-1,3)
        print("Identifying...")
        result = identify(self.catalog, obs, catalog_index=self.catalog_index)
        print("Result:", result)
        if result['best_solution']:
            R = result['best_solution']['R_est']
            print("R_est shape:", R.shape)
            # convert rotation matrix to quaternion
            M = np.eye(4)
            M[:3,:3] = R   # embed rotation into 4x4
            qx, qy, qz, qw = quaternion_from_matrix(M)
            
            q = Quaternion(x=qx, y=qy, z=qz, w=qw)
            self.publisher.publish(q)

def main(args=None):
    rclpy.init(args=args)
    node = IdentifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.')
    except Exception as e:
        node.get_logger().error(f'Unhandled exception: {e}')
        raise  # re‑raise if you want to see the traceback
    finally:
        # Always clean up
        node.destroy_node()
        rclpy.shutdown()