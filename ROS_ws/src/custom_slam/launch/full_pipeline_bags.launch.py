from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import xacro # <--- Add xacro here!

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    pkg_custom_slam = get_package_share_directory('custom_slam')
    rviz_config_path = os.path.join(pkg_custom_slam, 'config', 'rviz_detector.rviz')

    robot_fname = 'osr_full.urdf.xacro' # Change to 'osr.urdf.xacro' if that is your default
    xacro_file = os.path.join(pkg_custom_slam, 'urdf', robot_fname)
    robot_description = xacro.process_file(xacro_file).toxml()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': use_sim_time}
        ]
    )
    # --------------------------------------------------------

    # 1. SLAM Core 
    slam_node = Node(
        package='rover_slam',
        executable='slam_node',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 2. Geometric Matcher
    matcher = Node(
        package='custom_slam',
        executable='matcher_node',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 3. Path Visualizer
    odom_path = Node(
        package='custom_slam',
        executable='odom_to_path',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 4. Map Exporter
    exporter = Node(
        package='custom_slam',
        executable='map_exporter',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 5. RViz Visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        output='screen',
        arguments=['-d', rviz_config_path, '--ros-args', '--log-level', 'WARN'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        # robot_state_publisher_node, 
        slam_node,
        matcher,
        odom_path,
        exporter,
        rviz_node
    ])