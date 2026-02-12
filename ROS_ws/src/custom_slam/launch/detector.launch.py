from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import TimerAction
import os


def generate_launch_description():
    pkg_my_robot = get_package_share_directory('custom_slam')

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
    
    exporter = Node(
        package='custom_slam',
        executable='map_exporter',
    )
    # saver = Node(
    #     package='custom_slam',
    #     executable='image_saver',
    # )
    
    # Visualization (RViz) - Optional
    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz',
    #     output='screen',
    #     arguments=['-d', rviz_config_path]
    # )

    return LaunchDescription([
        robot,
        detector_node,
        yolo_node,
        exporter
        # saver
        # rviz_node
    ])
