from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare

import os
import xacro
from xacro import process_file

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')

    robot_fname = ['osr_full.urdf.xacro', 'osr.urdf.xacro']
    robot_version = 0

    rtab_remap=[
          ('/imu', '/imu_plugin/out'),
          ('rgb/image', '/color/image_raw'),
          ('rgb/camera_info', '/color/camera_info'),
          ('depth/image', '/aligned_depth_to_color/image_raw'),
          ('odom', '/osr/odom')] 

    # Correct paths
    # world = os.path.join(
    #     get_package_share_directory('custom_slam'),
    #     'worlds',
    #     'marsyard2022.world'
    # )

    pkg_gazebo = FindPackageShare("turtlebot3_gazebo").find("turtlebot3_gazebo")
    world = os.path.join(pkg_gazebo, "worlds", "turtlebot3_house.world")

    custom_slam_models_path = os.path.join(
        get_package_share_directory('custom_slam'),
        'models'
    )

    os.environ['GAZEBO_MODEL_PATH'] = os.environ.get('GAZEBO_MODEL_PATH', '') + ':' + custom_slam_models_path

    gazebo_dir = get_package_share_directory('gazebo_ros')

    rtabmap_param_file = PathJoinSubstitution([
        FindPackageShare('custom_slam'), 'config', 'custom_rtabmap_params.yaml'
    ])
    controller = Node(
            package='custom_slam',
            executable='controller_node',
            name='controller_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
            )
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world,
            'gui': 'true'
        }.items()
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


    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'turtlebot3',
            '-topic', 'robot_description',
            '-x', x_pose,
            '-y', y_pose,
            '-z', LaunchConfiguration('z_pose')
        ],
        output='screen'
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
    
    rgbd_sync = Node(
        package='rtabmap_sync',
        executable='rgbd_sync',
        name='rgbd_sync',
        output='screen',
        parameters=[rtabmap_param_file],
        remappings=rtab_remap

    )

    rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[rtabmap_param_file],
        arguments=['-d'],
        remappings=rtab_remap
    )
    viz = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=[rtabmap_param_file],
        remappings=rtab_remap
    )
    bev_node = Node(
            package='custom_slam',
            executable='bev_mapper_node',
            name='bev_mapper_node',
            output='screen',
            arguments=['--ros-args', '--log-level', 'custom_slam.bev_mapper_node:=info']
        )
    # odom_param_file = PathJoinSubstitution([
    #     FindPackageShare('custom_slam'), 'config', 'ekf_imu.yaml'
    # ])
    # odom = Node(
    #         package='robot_localization',
    #         executable='ekf_node',
    #         name='ekf_filter_node',
    #         output='screen',
    #         parameters=[odom_param_file]
    #     )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('x_pose', default_value='-3.0'),
        DeclareLaunchArgument('y_pose', default_value='-3.0'),
        DeclareLaunchArgument('z_pose', default_value='2.0'),
        gazebo,
        robot_state_publisher_node,
        spawn_after_gazebo,
        controller,
        load_joint_state_controller,
        rover_wheel_controller,
        servo_controller,
        # odom,
        rgbd_sync,
        rtabmap,
        viz,
        bev_node
    ])