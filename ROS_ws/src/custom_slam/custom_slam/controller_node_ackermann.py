import math
import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import TransformStamped, Quaternion
import tf2_ros


class Controller(Node):

    ROVER_WHEEL_RADIUS = 0.075  # You may want to adjust this to 0.05 from odometry node param
    WHEEL_BASE = 0.6  # Front to rear axel distance
    TRACK_WIDTH = 0.332 # left to right distance

    d1 = 0.177 # Distance from the center to the front-left wheel along the x-axis
    d2 = 0.310 # Distance from the center to the front-left wheel along the y-axis
    d3 = 0.274 # Distance from the center to the front-rear wheels along the y-axis
    d4 = 0.253 # Distance from the center to the center wheel along the x-axis

    def __init__(self):
        super().__init__('controller')

        self.declare_parameter('publish_tf', True)
        self.pub_tf = self.get_parameter('publish_tf').value

        self.motor_wheel_pub = self.create_publisher(Float64MultiArray, '/wheel_controller/commands', 1)
        self.servo_pub = self.create_publisher(JointTrajectory, '/servo_controller/joint_trajectory', 1)
        self.odom_pub = self.create_publisher(Odometry, '/calculated_odom', 10)

        self.sub = self.create_subscription(Twist, 'cmd_vel', self.msg_callback, 1)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu_plugin/out', self.imu_callback, 1)

        # Odometry state variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_positions = None
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
            wheel_indices = [
                msg.name.index("front_wheel_joint_left"),
                msg.name.index("front_wheel_joint_right"),
                msg.name.index("middle_wheel_joint_left"),
                msg.name.index("middle_wheel_joint_right"),
                msg.name.index("rear_wheel_joint_left"),
                msg.name.index("rear_wheel_joint_right"),
            ]
            servo_indices = [
                msg.name.index("front_wheel_joint_L"),
                msg.name.index("front_wheel_joint_R"),
                msg.name.index("rear_wheel_joint_L"),
                msg.name.index("rear_wheel_joint_R"),
                ]
        except ValueError:
            self.get_logger().warn("Wheel joint names not found in joint_states")
            return

        positions = np.array([msg.position[i] for i in wheel_indices])
        servo_positions = np.array([msg.position[i] for i in servo_indices])
        current_time = rclpy.time.Time.from_msg(msg.header.stamp)       
         
        if self.last_positions is None:
            self.last_positions = positions.copy()
            self.last_time = current_time
            return

        dt = (current_time - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            self.get_logger().debug(f"Skipping joint_state update because dt={dt}")
            return


        # Wheel displacements
        delta_pos = positions - self.last_positions
        wheel_displacements = delta_pos * self.ROVER_WHEEL_RADIUS

        
        # Approx angular displacement using left vs right side wheels
        left_disp = np.mean([wheel_displacements[0], wheel_displacements[2], wheel_displacements[4]])
        right_disp = np.mean([wheel_displacements[1], wheel_displacements[3], wheel_displacements[5]])
        
        
        delta_f = 0.5 * (servo_positions[0] + servo_positions[1])
        delta_r = 0.5 * (servo_positions[2] + servo_positions[3])

        # Forward displacement from encoders
        d_center = (right_disp + left_disp) / 2.0
        
        if abs(d_center) < 0.001 and abs(right_disp - left_disp) > 0.001:
            # Spot turn → use differential-drive kinematics
            d_theta = (right_disp - left_disp) / self.TRACK_WIDTH
            R = 0.0
            dx = 0.0
            dy = 0.0
        else:
            # Normal motion → use steering geometry
            tan_diff = math.tan(delta_f) - math.tan(delta_r)
            if abs(tan_diff) < 1e-6:
                R = float('inf')
                d_theta = 0.0
                dx = d_center * math.cos(self.theta)
                dy = d_center * math.sin(self.theta)
                
            else:
                R = self.WHEEL_BASE / tan_diff
                d_theta = d_center / R
                dx = R * (math.sin(self.theta + d_theta) - math.sin(self.theta))
                dy = -R * (math.cos(self.theta + d_theta) - math.cos(self.theta))

        self.x += dx
        self.y += dy
        self.theta = math.atan2(math.sin(self.theta + d_theta), math.cos(self.theta + d_theta))
        
        
                
        turning_factor = abs(d_theta)
        # Base covariances
        xy_cov = 0.001  # 1mm uncertainty in straight motion
        yaw_cov = 0.01  # Base rotational uncertainty
        
        # Increase uncertainty when turning
        if turning_factor > 0.01:  # If turning
            xy_cov = 0.005  # 5mm uncertainty
            yaw_cov = 0.1   # 10x worse yaw uncertainty
        
        # Pose covariance (6x6 matrix, flattened to 36 elements)
        # Order: x, y, z, roll, pitch, yaw
        self.cov_pose = [
            xy_cov, 0.0,    0.0,    0.0,    0.0,    0.0,     # x
            0.0,    xy_cov, 0.0,    0.0,    0.0,    0.0,     # y
            0.0,    0.0,    1e6,    0.0,    0.0,    0.0,     # z (unused)
            0.0,    0.0,    0.0,    1e6,    0.0,    0.0,     # roll (unused)
            0.0,    0.0,    0.0,    0.0,    1e6,    0.0,     # pitch (unused)
            0.0,    0.0,    0.0,    0.0,    0.0,    yaw_cov  # yaw
        ]
        
        # Twist covariance (6x6 matrix)
        vel_cov = 0.01
        vyaw_cov = 0.1 if turning_factor > 0.01 else 0.01
        
        self.cov_twist = [
            vel_cov, 0.0,     0.0,     0.0,     0.0,     0.0,
            0.0,     vel_cov, 0.0,     0.0,     0.0,     0.0,
            0.0,     0.0,     1e6,     0.0,     0.0,     0.0,
            0.0,     0.0,     0.0,     1e6,     0.0,     0.0,
            0.0,     0.0,     0.0,     0.0,     1e6,     0.0,
            0.0,     0.0,     0.0,     0.0,     0.0,     vyaw_cov
        ]



        # Publish odom
        self.publish_odom(current_time, dx/dt, dy/dt, d_theta/dt)
        
        if self.pub_tf:
            self.publish_tf(current_time)

        # Update memory
        self.last_positions = positions
        self.last_time = current_time


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
        
        odom_msg.pose.covariance = self.cov_pose
        odom_msg.twist.covariance = self.cov_twist
        
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
        # self.theta = yaw

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
            self.FR_servo_data = theta_front_farthest
            self.RL_servo_data = -theta_front_closest
            self.RR_servo_data = -theta_front_farthest
        else:
            self.FL_servo_data = -theta_front_farthest
            self.FR_servo_data = -theta_front_closest
            self.RL_servo_data = theta_front_farthest
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

        if l < 0:
            self.FL_data = ang_vel_corner_closest
            self.FR_data = ang_vel_corner_farthest

            self.ML_data = ang_vel_middle_closest
            self.MR_data = ang_vel_middle_farthest

            self.RL_data = ang_vel_corner_closest
            self.RR_data = ang_vel_corner_farthest
        else:
            self.FL_data = ang_vel_corner_farthest
            self.FR_data = ang_vel_corner_closest

            self.ML_data = ang_vel_middle_farthest
            self.MR_data = ang_vel_middle_closest

            self.RL_data = ang_vel_corner_farthest
            self.RR_data = ang_vel_corner_closest

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
        self.FL_data = math.sqrt(self.d1 ** 2 + self.d3 ** 2) * msg.angular.z / self.ROVER_WHEEL_RADIUS
        self.RL_data = math.sqrt(self.d1 ** 2 + self.d2 ** 2) * msg.angular.z / self.ROVER_WHEEL_RADIUS

        self.ML_data = self.d4 * msg.angular.z / self.ROVER_WHEEL_RADIUS
        self.FR_data = -math.sqrt(self.d1 ** 2 + self.d3 ** 2) * msg.angular.z / self.ROVER_WHEEL_RADIUS

        self.RR_data = -self.d4 * msg.angular.z / self.ROVER_WHEEL_RADIUS
        self.MR_data = -math.sqrt(self.d1 ** 2 + self.d2 ** 2) * msg.angular.z / self.ROVER_WHEEL_RADIUS

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
