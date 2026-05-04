import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('yolo_realsense')
    default_rviz_config = os.path.join(package_share, 'rviz', 'd435_color.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('width', default_value='640'),
        DeclareLaunchArgument('height', default_value='480'),
        DeclareLaunchArgument('fps', default_value='30'),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz_config),
        Node(
            package='yolo_realsense',
            executable='d435_color_publisher',
            name='d435_color_publisher',
            output='screen',
            parameters=[{
                'width': LaunchConfiguration('width'),
                'height': LaunchConfiguration('height'),
                'fps': LaunchConfiguration('fps'),
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(LaunchConfiguration('start_rviz')),
            output='screen',
        ),
    ])
