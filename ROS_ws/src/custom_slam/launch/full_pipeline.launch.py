from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import TimerAction
import os


def generate_launch_description():
    pkg_my_robot = get_package_share_directory('custom_slam')
    rviz_config_path = os.path.join(pkg_my_robot, 'config', 'rviz_detector.rviz')
    stereo_launch_path = os.path.join(
        get_package_share_directory('stereo_image_proc'),
        'launch',
        'stereo_image_proc.launch.py'  # Note: The file name might vary slightly, but this is standard
    )
    # Robot + Gazebo world
    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_my_robot, 'launch', 'simulation.launch.py')
        )
    )

    detector_node = Node(
        package='rover_perception',
        executable='rock_3d_localizer'
    )

    yolo_node = Node(
        package='rover_perception',
        executable='yolo_detector',
    )
    
    perception_node = Node(
        package='rover_perception',
        executable='integrated_rock_perception',
        parameters=[{'use_sim_time': True}]
    )
    
    exporter = Node(
        package='custom_slam',
        executable='map_exporter',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        output='screen',
        arguments=['-d', rviz_config_path, '--ros-args', '--log-level', 'WARN'],
        parameters=[{'use_sim_time': True}],
    )
    
    matcher = Node(
        package='custom_slam',
        executable='matcher_node',
        parameters=[{'use_sim_time': True}]
    )
    slam_node = Node(
        package='rover_slam',
        executable='slam_node',
        parameters=[{'use_sim_time': True}]
    )
    
    rect = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(stereo_launch_path),
            launch_arguments={
                'approximate_sync': 'True',  # Important for cameras that don't trigger perfectly together
                'left_namespace': '/camera/left',
                'right_namespace': '/camera/right',
                'target_frame_id': 'base_link', # Or 'odom' or 'camera_left_optical_frame'
            }.items()
        )

    return LaunchDescription([
        robot,
        # detector_node,
        # yolo_node,
        perception_node,
        exporter,
        matcher,
        rviz_node,
        rect,
        # slam_node
    ])
