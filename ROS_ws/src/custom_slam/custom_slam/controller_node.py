import math
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import TransformStamped, Quaternion
import tf2_ros


class Controller(Node):

    ROVER_WHEEL_RADIUS = 0.068  # You may want to adjust this to 0.05 from odometry node param
    WHEEL_BASE = 0.332  # Use wheel_base param from odometry node

    d1 = 0.177
    d2 = 0.310
    d3 = 0.274
    d4 = 0.253

    def __init__(self):
        super().__init__('controller')

        self.motor_wheel_pub = self.create_publisher(Float64MultiArray, '/wheel_controller/commands', 1)
        self.servo_pub = self.create_publisher(JointTrajectory, '/servo_controller/joint_trajectory', 1)
        self.odom_pub = self.create_publisher(Odometry, 'osr/odom', 10)

        self.sub = self.create_subscription(Twist, 'cmd_vel', self.msg_callback, 1)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu_plugin/out', self.imu_callback, 1)

        # Odometry state variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_left_pos = None
        self.last_right_pos = None
        self.last_time = None

        # Other Controller variables
        self.servo_theta = 0.0
        self.l = 0.0

        self.FL_data = 0.0
        self.FR_data = 0.0
        self.ML_data = 0.0
        self.MR_data = 0.0
        self.RL_data = 0.0
        self.RR_data = 0.0

        self.FL_servo_data = 0.0
        self.FR_servo_data = 0.0
        self.RL_servo_data = 0.0
        self.RR_servo_data = 0.0

        self.delay_ = True
        self.pri_velocity = Twist()

        self.br = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info('Controller node started with integrated odometry.')

    def joint_state_callback(self, msg: JointState):
        # Extract left and right wheel positions by joint name
        try:
            left_index = msg.name.index('front_wheel_joint_left')
            right_index = msg.name.index('front_wheel_joint_right')
        except ValueError:
            self.get_logger().warning('Expected joint names not found in joint_states.')
            return

        left_pos = msg.position[left_index]
        right_pos = msg.position[right_index]
        now = self.get_clock().now()

        if self.last_left_pos is None or self.last_right_pos is None:
            self.last_left_pos = left_pos
            self.last_right_pos = right_pos
            self.last_time = now
            return

        dt = (now - self.last_time).nanoseconds * 1e-9  # convert ns to seconds
        if dt == 0:
            return

        # Wheel displacements
        d_left = (left_pos - self.last_left_pos) * self.ROVER_WHEEL_RADIUS
        d_right = (right_pos - self.last_right_pos) * self.ROVER_WHEEL_RADIUS

        self.last_left_pos = left_pos
        self.last_right_pos = right_pos
        self.last_time = now

        # Compute odometry increments
        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / self.WHEEL_BASE

        if d_theta == 0:
            dx = d_center * math.cos(self.theta)
            dy = d_center * math.sin(self.theta)
        else:
            radius = d_center / d_theta
            dx = radius * (math.sin(self.theta + d_theta) - math.sin(self.theta))
            dy = -radius * (math.cos(self.theta + d_theta) - math.cos(self.theta))

        self.x += dx
        self.y += dy
        self.theta = (self.theta + d_theta) % (2 * math.pi)

        # Compute velocities
        vx = dx / dt
        vy = dy / dt
        vtheta = d_theta / dt

        self.publish_odom(now, vx, vy, vtheta)
        self.publish_tf(now)

        # Keep your existing velocity tracking if needed here or separate

    def publish_odom(self, now, vx, vy, vtheta):
        odom_msg = Odometry()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_footprint'

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        quat = quaternion_from_euler(0, 0, self.theta)
        odom_msg.pose.pose.orientation = Quaternion(
            x=quat[0], y=quat[1], z=quat[2], w=quat[3]
        )

        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.linear.y = vy
        odom_msg.twist.twist.angular.z = vtheta

        self.odom_pub.publish(odom_msg)

    def publish_tf(self, now):
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        quat = quaternion_from_euler(0, 0, self.theta)
        t.transform.rotation = Quaternion(
            x=quat[0], y=quat[1], z=quat[2], w=quat[3]
        )

        self.br.sendTransform(t)


    def imu_callback(self, msg: Imu):
        # Convert quaternion to yaw angle (theta)
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]
        # Use tf_transformations or tf2 to convert to roll, pitch, yaw
        import tf_transformations
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)
        self.theta = yaw

    def msg_callback(self, msg: Twist):
        if (self.pri_velocity.linear.x == msg.linear.x and
                self.pri_velocity.angular.z == msg.angular.z):
            return

        if self.delay_:
            self.delay_ = False

            if msg.angular.z == 0 and msg.linear.x != 0:
                self.go_straight(msg)
                self.publish_velocity()
                self.publish_angles()

            elif msg.angular.z != 0 and msg.linear.x == 0:
                self.FL_servo_data = -math.atan(self.d3 / self.d1)
                self.FR_servo_data = math.atan(self.d3 / self.d1)
                self.RL_servo_data = math.atan(self.d2 / self.d1)
                self.RR_servo_data = -math.atan(self.d2 / self.d1)

                self.rotate_in_place(msg)
                self.publish_angles()
                self.publish_velocity()

            elif msg.angular.z != 0 and msg.linear.x != 0:
                self.twist_to_turning_radius(msg)
                self.calculate_servo_angle(self.l)
                self.calculate_drive_velocity(msg.linear.x, self.l)
                self.publish_angles()
                self.publish_velocity()

            elif msg.angular.z == 0 and msg.linear.x == 0:
                self.stop()
                self.publish_angles()
                self.publish_velocity()

            self.delay_ = True

        self.pri_velocity.linear.x = msg.linear.x
        self.pri_velocity.angular.z = msg.angular.z

    def calculate_servo_angle(self, l):
        theta_front_closest = math.atan2(self.d3, abs(l) - self.d1)
        theta_front_farthest = math.atan2(self.d3, abs(l) + self.d1)

        if l > 0:
            self.FL_servo_data = theta_front_closest
            self.FR_servo_data = theta_front_closest
            self.RL_servo_data = -theta_front_closest
            self.RR_servo_data = -theta_front_closest
        else:
            self.FL_servo_data = -theta_front_closest
            self.FR_servo_data = -theta_front_closest
            self.RL_servo_data = theta_front_closest
            self.RR_servo_data = theta_front_closest

    def calculate_drive_velocity(self, velocity, l):
        angular_velocity_center = velocity / abs(l)

        vel_middle_closest = (abs(l) - self.d4) * angular_velocity_center
        vel_corner_closest = math.hypot(abs(l) - self.d1, self.d3) * angular_velocity_center
        vel_corner_farthest = math.hypot(abs(l) + self.d1, self.d3) * angular_velocity_center
        vel_middle_farthest = (abs(l) + self.d4) * angular_velocity_center

        ang_vel_middle_closest = vel_middle_closest / self.ROVER_WHEEL_RADIUS
        ang_vel_corner_closest = vel_corner_closest / self.ROVER_WHEEL_RADIUS
        ang_vel_corner_farthest = vel_corner_farthest / self.ROVER_WHEEL_RADIUS
        ang_vel_middle_farthest = vel_middle_farthest / self.ROVER_WHEEL_RADIUS

        if l > 0:
            self.FL_data = ang_vel_corner_closest
            self.RL_data = ang_vel_corner_closest

            self.ML_data = ang_vel_middle_closest
            self.FR_data = ang_vel_corner_farthest

            self.RR_data = ang_vel_corner_farthest
            self.MR_data = ang_vel_middle_farthest
        else:
            self.FL_data = ang_vel_corner_farthest
            self.RL_data = ang_vel_corner_farthest

            self.ML_data = ang_vel_middle_farthest
            self.FR_data = ang_vel_corner_closest

            self.RR_data = ang_vel_corner_closest
            self.MR_data = ang_vel_middle_closest

    def stop(self):
        self.FL_data = 0.0
        self.RL_data = 0.0
        self.ML_data = 0.0
        self.FR_data = 0.0
        self.RR_data = 0.0
        self.MR_data = 0.0

        self.FL_servo_data = 0.0
        self.FR_servo_data = 0.0
        self.RL_servo_data = 0.0
        self.RR_servo_data = 0.0

    def go_straight(self, msg: Twist):
        velocity_data = msg.linear.x / self.ROVER_WHEEL_RADIUS

        self.FL_data = velocity_data
        self.FR_data = velocity_data
        self.ML_data = velocity_data
        self.MR_data = velocity_data
        self.RL_data = velocity_data
        self.RR_data = velocity_data

        self.FL_servo_data = 0.0
        self.FR_servo_data = 0.0
        self.RL_servo_data = 0.0
        self.RR_servo_data = 0.0

    def twist_to_turning_radius(self, msg: Twist):
        self.l = msg.linear.x / msg.angular.z

    def rotate_in_place(self, msg: Twist):
        self.FL_data = -math.sqrt(self.d1 ** 2 + self.d3 ** 2) * msg.angular.z / self.ROVER_WHEEL_RADIUS
        self.RL_data = -math.sqrt(self.d1 ** 2 + self.d2 ** 2) * msg.angular.z / self.ROVER_WHEEL_RADIUS

        self.ML_data = -self.d4 * msg.angular.z / self.ROVER_WHEEL_RADIUS
        self.FR_data = math.sqrt(self.d1 ** 2 + self.d3 ** 2) * msg.angular.z / self.ROVER_WHEEL_RADIUS

        self.RR_data = self.d4 * msg.angular.z / self.ROVER_WHEEL_RADIUS
        self.MR_data = math.sqrt(self.d1 ** 2 + self.d2 ** 2) * msg.angular.z / self.ROVER_WHEEL_RADIUS

    def publish_velocity(self):
        wheel_msg = Float64MultiArray()
        wheel_msg.data = [self.ML_data, self.MR_data,
                          self.FL_data, self.FR_data,
                          self.RL_data, self.RR_data]
        self.motor_wheel_pub.publish(wheel_msg)

    def publish_angles(self):
        servo = JointTrajectory()
        servo.joint_names = ['front_wheel_joint_R', 'front_wheel_joint_L', 'rear_wheel_joint_R', 'rear_wheel_joint_L']

        point = JointTrajectoryPoint()
        point.positions = [self.FR_servo_data, self.FL_servo_data, self.RR_servo_data, self.RL_servo_data]
        point.velocities = [0.0, 0.0, 0.0, 0.0]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.2 * 1e9)

        servo.points.append(point)

        self.servo_pub.publish(servo)


def main(args=None):
    rclpy.init(args=args)
    node = Controller()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
