from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

import os

def launch_setup(context, *args, **kwargs):
    if not 'TURTLEBOT3_MODEL' in os.environ:
        os.environ['TURTLEBOT3_MODEL'] = 'waffle'

    # Directories
    pkg_turtlebot3_gazebo = get_package_share_directory(
        'turtlebot3_gazebo')
    pkg_nav2_bringup = get_package_share_directory(
        'nav2_bringup')
    pkg_rtabmap_demos = get_package_share_directory(
        'rtabmap_demos')
    custom_rviz_config = PathJoinSubstitution(
        [FindPackageShare('custom_slam'), 'rviz', 'rviz_config.rviz']
    )
    world = LaunchConfiguration('world').perform(context)
    
    nav2_params_file = PathJoinSubstitution(
        [FindPackageShare('rtabmap_demos'), 'params', 'turtlebot3_rgbd_nav2_params.yaml']
    )

    # Paths
    gazebo_launch = PathJoinSubstitution(
        [pkg_turtlebot3_gazebo, 'launch', f'turtlebot3_{world}.launch.py'])
    nav2_launch = PathJoinSubstitution(
        [pkg_nav2_bringup, 'launch', 'navigation_launch.py'])
    rviz_launch = PathJoinSubstitution(
        [pkg_nav2_bringup, 'launch', 'rviz_launch.py'])
    rtabmap_launch = PathJoinSubstitution(
        [pkg_rtabmap_demos, 'launch', 'turtlebot3', 'turtlebot3_rgbd.launch.py'])

    # Includes
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gazebo_launch]),
        launch_arguments=[
            ('x_pose', LaunchConfiguration('x_pose')),
            ('y_pose', LaunchConfiguration('y_pose')),
            ('ros_args', '--log-level error')
        ]
    )
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch]),
        launch_arguments=[
            ('use_sim_time', 'true'),
            ('params_file', nav2_params_file),
            ('ros_args', '--log-level error')
        ]
    )
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rviz_launch]),
        launch_arguments=[
            ('ros_args', '--log-level error'),
            ('rviz_config', custom_rviz_config)
        ]
    )
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rtabmap_launch]),
        launch_arguments=[
            ('localization', LaunchConfiguration('localization')),
            ('use_sim_time', 'true'),
            ('ros_args', '--log-level error')
        ]
    )
    return [
        # Nodes to launch
        nav2,
        rviz,
        rtabmap,
        gazebo
    ]

def generate_launch_description():
    # gazebo_launch_file = os.path.join(
    #     get_package_share_directory('gazebo_ros'),
    #     'launch',
    #     'gazebo.launch.py'
    # )

    bev_node = Node(
            package='custom_slam',
            executable='bev_mapper_node',
            name='bev_mapper_node',
            output='screen',
        )

    return LaunchDescription([
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(gazebo_launch_file)
        # ),
        bev_node,
        # Launch arguments
        DeclareLaunchArgument(
            'localization', default_value='false',
            description='Launch in localization mode.'),

        DeclareLaunchArgument(
            'world', default_value='house',
            choices=['world', 'house', 'dqn_stage1', 'dqn_stage2', 'dqn_stage3', 'dqn_stage4'],
            description='Turtlebot3 gazebo world.'),
        
        DeclareLaunchArgument(
            'x_pose', default_value='-2.0',
            description='Initial position of the robot in the simulator.'),
        
        DeclareLaunchArgument(
            'y_pose', default_value='0.5',
            description='Initial position of the robot in the simulator.'),

        OpaqueFunction(function=launch_setup)
    ])
