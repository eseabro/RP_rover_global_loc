from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit

import os
import xacro
from xacro import process_file

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')

    robot_fname = ['osr_full.urdf.xacro', 'osr.urdf.xacro']
    robot_version = 0


    gz_sim_pkg = get_package_share_directory('ros_gz_sim')

    # Correct paths
    world = os.path.join(
        get_package_share_directory('leo_gz_worlds'),
        'worlds',
        'marsyard2021.sdf'
    )
    # world = os.path.join(
    #     get_package_share_directory('custom_slam'),
    #     'worlds',
    #     'cnes_marsyard.sdf'
    # )

    ## Gazebo Harmonic

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            # gz_args is a single string: <world> plus any CLI flags
            'gz_args': f'{world} -r', # add -v4 for verbose
            'gui': 'true'
        }.items()
    )


    custom_slam_models_path = os.path.join(
        get_package_share_directory('custom_slam'),
        'models'
    )

    os.environ['GZ_SIM_RESOURCE_PATH'] = os.environ.get('GZ_SIM_RESOURCE_PATH', '') + ':' + custom_slam_models_path

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'osr_rover',
            '-x', x_pose,
            '-y', y_pose,
            '-z', LaunchConfiguration('z_pose')
        ],
        output='screen'
    )
    
    gz_bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    parameters=[{'config_file': os.path.join(
        get_package_share_directory('custom_slam'),
        'config',
        'gz_bridge.yaml'
    )}, 
    {'use_sim_time': use_sim_time}]
    )
    
    controller = Node(
            package='custom_slam',
            executable='controller_node_ackermann',
            name='controller_node_ackermann',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time,
                         'publish_tf': False}]
            )

    xacro_file = os.path.join(get_package_share_directory('custom_slam'),
                              'urdf',
                              robot_fname[robot_version])

    robot_description = process_file(xacro_file).toxml()

    # robot_description_content = process_file(robot_urdf_path).toxml()

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
    odom_path = Node(
        package='custom_slam',
        executable='odom_to_path'
    )




    spawn_after_gazebo = TimerAction(
        period=5.0,  # seconds, adjust as needed
        actions=[spawn]
    )

    load_joint_state_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen'
    )

    # wheel_velocity_controller
    rover_wheel_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'wheel_controller'],
        output='screen'
    )

    # servo_controller
    servo_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'servo_controller'],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('x_pose', default_value='-0.0'),
        DeclareLaunchArgument('y_pose', default_value='-0.0'),
        DeclareLaunchArgument('z_pose', default_value='2.0'),
        gazebo,
        gz_bridge,
        robot_state_publisher_node,
        spawn_after_gazebo,
        controller,
        odom_path,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn,
                on_exit=[load_joint_state_controller],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_controller,
                on_exit=[rover_wheel_controller],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=rover_wheel_controller,
                on_exit=[servo_controller],
            )
        )
    ])