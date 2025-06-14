from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from xacro import process_file  # Make sure `xacro` is installed

import os
import xacro

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')

    # Correct paths
    world = os.path.join(
        get_package_share_directory('custom_slam'),
        'worlds',
        'marsyard2022.world'
    )
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
            output='screen'
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
                              'osr.urdf.xacro')

    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    # robot_description_content = process_file(robot_urdf_path).toxml()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': doc.toxml()},
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
    controller_manager = Node(
            package='controller_manager',
            executable='spawner',
            arguments=['wheel_controller', '--controller-manager', '/controller_manager'],
            output='screen',
        )
    
    rgbd_sync = Node(
        package='rtabmap_sync',
        executable='rgbd_sync',
        name='rgbd_sync',
        output='screen',
        parameters=[rtabmap_param_file],
        remappings=[
            ('rgb/image', '/camera/color/image_raw'),
            ('depth/image', '/camera/depth/image_raw'),
            ('rgb/camera_info', '/camera/color/camera_info'),
            ('odom', '/odom')
        ]
    )

    rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[rtabmap_param_file],
        arguments=['-d'],
        remappings=[
            ('rgb/image', '/camera/color/image_raw'),
            ('depth/image', '/camera/depth/image_raw'),
            ('rgb/camera_info', '/camera/color/camera_info'),
            ('odom', '/odom')
        ]
    )
    viz = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=[{
            'frame_id': 'base_footprint',
            'odom_frame_id': 'odom',
            'use_sim_time': use_sim_time,
        }]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        DeclareLaunchArgument('z_pose', default_value='10.0'),
        gazebo,
        robot_state_publisher_node,
        spawn_after_gazebo,
        controller,
        controller_manager,
        rgbd_sync,
        rtabmap,
        viz
    ])
