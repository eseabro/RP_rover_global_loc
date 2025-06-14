import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster
import numpy as np

ROVER_WHEEL_RADIUS = 0.075
d1 = 0.177
d2 = 0.310
d3 = 0.274
d4 = 0.253

class OSRController(Node):
    def __init__(self):
        super().__init__('controller')

        # Publishers
        self.motor_wheel_pub = self.create_publisher(Float64MultiArray, '/wheel_controller/commands', 1)
        self.servo_pub = self.create_publisher(JointTrajectory, '/servo_controller/joint_trajectory', 1)
        self.odom_pub = self.create_publisher(Odometry, 'osr/odom', 10)
        self.br = TransformBroadcaster(self)

        # Subscriptions
        self.sub = self.create_subscription(Twist, 'cmd_vel', self.msg_callback, 1)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 1)
        self.imu_sub = self.create_subscription(Imu, 'imu_plugin/out', self.imu_callback, 1)

        # State
        self.pri_velocity = Twist()
        self.theta = 0.0
        self.x_position = 0.0
        self.y_position = 0.0
        self.pre_dl = 0.0

        self.FL_data = self.FR_data = self.ML_data = self.MR_data = self.RL_data = self.RR_data = 0.0
        self.FL_servo_data = self.FR_servo_data = self.RL_servo_data = self.RR_servo_data = 0.0
        self.fl_vel = self.fr_vel = self.ml_vel = self.mr_vel = self.rl_vel = self.rr_vel = 0.0

    def joint_state_callback(self, msg):
        try:
            self.fl_vel = msg.velocity[msg.name.index('front_wheel_joint_left')]
            self.fr_vel = msg.velocity[msg.name.index('front_wheel_joint_right')]
            self.ml_vel = msg.velocity[msg.name.index('middle_wheel_joint_left')]
            self.mr_vel = msg.velocity[msg.name.index('middle_wheel_joint_right')]
            self.rl_vel = msg.velocity[msg.name.index('rear_wheel_joint_left')]
            self.rr_vel = msg.velocity[msg.name.index('rear_wheel_joint_right')]
        except (ValueError, IndexError) as e:
            self.get_logger().warn(f"JointState missing expected joint names or velocities: {str(e)}")
            return

        self.publish_odometry()


    def imu_callback(self, msg):
        orientation_q = msg.orientation
        _, _, yaw = euler_from_quaternion([
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        ])
        self.theta = yaw

    def publish_odometry(self):
        current_dl = (self.fl_vel + self.fr_vel + self.ml_vel + self.mr_vel + self.rl_vel + self.rr_vel) * ROVER_WHEEL_RADIUS / 6
        dl = current_dl - self.pre_dl
        self.pre_dl = current_dl

        self.x_position += dl * math.cos(self.theta)
        self.y_position += dl * math.sin(self.theta)

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_footprint"
        odom_msg.pose.pose.position.x = self.x_position
        odom_msg.pose.pose.position.y = self.y_position

        q = quaternion_from_euler(0, 0, self.theta)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        # Publish TF
        t = TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.translation.x = self.x_position
        t.transform.translation.y = self.y_position
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.br.sendTransform(t)
        self.odom_pub.publish(odom_msg)

    def msg_callback(self, msg):
        if (self.pri_velocity.linear.x == msg.linear.x and self.pri_velocity.angular.z == msg.angular.z):
            return

        if msg.angular.z == 0.0 and msg.linear.x != 0.0:
            self.go_straight(msg)
        elif msg.angular.z != 0.0 and msg.linear.x == 0.0:
            self.set_servo_angles_for_rotation()
            self.rotate_in_place(msg)
        elif msg.angular.z != 0.0 and msg.linear.x != 0.0:
            l = msg.linear.x / msg.angular.z
            self.calculate_servo_angle(l)
            self.calculate_drive_velocity(msg.linear.x, l)
        else:
            self.stop()

        self.publish_angles()
        self.publish_velocity()

        self.pri_velocity.linear.x = msg.linear.x
        self.pri_velocity.angular.z = msg.angular.z

    def calculate_servo_angle(self, l):
        theta = math.atan2(d3, abs(l) - d1)
        sign = 1 if l > 0 else -1
        self.FL_servo_data = sign * theta
        self.FR_servo_data = sign * theta
        self.RL_servo_data = -sign * theta
        self.RR_servo_data = -sign * theta

    def calculate_drive_velocity(self, velocity, l):
        omega = velocity / abs(l)
        radius_m_cl = abs(l) - d4
        radius_c_cl = math.hypot(abs(l) - d1, d3)
        radius_c_fa = math.hypot(abs(l) + d1, d3)
        radius_m_fa = abs(l) + d4

        if l > 0:
            self.FL_data = omega * radius_c_cl / ROVER_WHEEL_RADIUS
            self.FR_data = omega * radius_c_fa / ROVER_WHEEL_RADIUS
            self.ML_data = omega * radius_m_cl / ROVER_WHEEL_RADIUS
            self.MR_data = omega * radius_m_fa / ROVER_WHEEL_RADIUS
            self.RL_data = self.FL_data
            self.RR_data = self.FR_data
        else:
            self.FR_data = omega * radius_c_cl / ROVER_WHEEL_RADIUS
            self.FL_data = omega * radius_c_fa / ROVER_WHEEL_RADIUS
            self.MR_data = omega * radius_m_cl / ROVER_WHEEL_RADIUS
            self.ML_data = omega * radius_m_fa / ROVER_WHEEL_RADIUS
            self.RR_data = self.FR_data
            self.RL_data = self.FL_data

    def stop(self):
        self.FL_data = self.FR_data = self.ML_data = self.MR_data = self.RL_data = self.RR_data = 0.0
        self.FL_servo_data = self.FR_servo_data = self.RL_servo_data = self.RR_servo_data = 0.0

    def go_straight(self, msg):
        vel = msg.linear.x / ROVER_WHEEL_RADIUS
        self.FL_data = self.FR_data = self.ML_data = self.MR_data = self.RL_data = self.RR_data = vel
        self.FL_servo_data = self.FR_servo_data = self.RL_servo_data = self.RR_servo_data = 0.0

    def set_servo_angles_for_rotation(self):
        self.FL_servo_data = -math.atan2(d3, d1)
        self.FR_servo_data = math.atan2(d3, d1)
        self.RL_servo_data = math.atan2(d2, d1)
        self.RR_servo_data = -math.atan2(d2, d1)

    def rotate_in_place(self, msg):
        ang_z = msg.angular.z
        self.FL_data = -math.hypot(d1, d3) * ang_z / ROVER_WHEEL_RADIUS
        self.FR_data = math.hypot(d1, d3) * ang_z / ROVER_WHEEL_RADIUS
        self.ML_data = -d4 * ang_z / ROVER_WHEEL_RADIUS
        self.MR_data = d4 * ang_z / ROVER_WHEEL_RADIUS
        self.RL_data = -math.hypot(d1, d2) * ang_z / ROVER_WHEEL_RADIUS
        self.RR_data = math.hypot(d1, d2) * ang_z / ROVER_WHEEL_RADIUS

    def publish_velocity(self):
        msg = Float64MultiArray()
        msg.data = [
            self.ML_data, self.MR_data,
            self.FL_data, self.FR_data,
            self.RL_data, self.RR_data
        ]
        self.motor_wheel_pub.publish(msg)

    def publish_angles(self):
        servo = JointTrajectory()
        servo.joint_names = [
            "front_wheel_joint_R", "front_wheel_joint_L",
            "rear_wheel_joint_R", "rear_wheel_joint_L"
        ]
        point = JointTrajectoryPoint()
        point.positions = [
            self.FR_servo_data, self.FL_servo_data,
            self.RR_servo_data, self.RL_servo_data
        ]
        point.velocities = [0.0] * 4
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.2 * 1e9)
        servo.points.append(point)
        self.servo_pub.publish(servo)

def main(args=None):
    rclpy.init(args=args)
    node = OSRController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
