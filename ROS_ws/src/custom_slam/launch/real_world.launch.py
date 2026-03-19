from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():
    pkg_my_robot = get_package_share_directory('custom_slam')
    rviz_config_path = os.path.join(pkg_my_robot, 'config', 'rviz_detector.rviz')
    
    # CRITICAL: Forces all nodes to use the timestamp from the rosbag, not your computer's clock!
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # ══════════════════════════════════════════════════════════════════════
    # 1. HARDWARE TFs (Extracted directly from your calib.yaml)
    # The bag file does not have Gazebo or a URDF to broadcast where the 
    # cameras are. We must broadcast the IMU -> Camera transforms statically.
    # ══════════════════════════════════════════════════════════════════════
    # Cam 0 (Left): x=0.0668, y=0.0701, z=0.0107 | Yaw=-1.5708, Pitch=0, Roll=-1.5708
    tf_imu_to_cam0 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_imu_to_cam0',
        # arguments: x, y, z, yaw, pitch, roll, frame_id, child_frame_id
        arguments=['0.0668', '0.0701', '0.0107', '-1.5708', '0.0', '-1.5708', 'imu_link', 'cam_0_optical'],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Cam 1 (Right): x=0.0667, y=-0.0558, z=0.0106 | Yaw=-1.5708, Pitch=0, Roll=-1.5708
    tf_imu_to_cam1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_imu_to_cam1',
        arguments=['0.0667', '-0.0558', '0.0106', '-1.5708', '0.0', '-1.5708', 'imu_link', 'cam_1_optical'],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # ══════════════════════════════════════════════════════════════════════
    # 2. THE CUSTOM FRONT-END (Replacing stereo_image_proc)
    # ══════════════════════════════════════════════════════════════════════
    # Standard stereo_image_proc WILL CRASH on your double_sphere fisheye images.
    # You must replace it with a custom node that uses OpenCV omnidir to undistort
    # the images and generate the PointCloud2.
    custom_stereo_rectifier = Node(
        package='custom_slam', # Replace with wherever you put your custom undistort node
        executable='fisheye_undistort_node', 
        name='fisheye_stereo',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ══════════════════════════════════════════════════════════════════════
    # 3. YOUR EXISTING BACKEND PIPELINE
    # ══════════════════════════════════════════════════════════════════════
    perception_node = Node(
        package='rover_perception',
        executable='integrated_rock_perception',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        output='screen',
        arguments=['-d', rviz_config_path, '--ros-args', '--log-level', 'WARN'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        tf_imu_to_cam0,
        tf_imu_to_cam1,
        custom_stereo_rectifier,
        perception_node,
        rviz_node
    ])