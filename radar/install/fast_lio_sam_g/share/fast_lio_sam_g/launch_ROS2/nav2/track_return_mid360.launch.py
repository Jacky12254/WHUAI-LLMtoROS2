from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_recorded_topic = "/track_return/recorded_path"
    default_return_topic = "/track_return/return_path"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "odom_topic",
                default_value="/OdometryHighFreq",
                description="Odometry topic used for path recording.",
            ),
            DeclareLaunchArgument(
                "frame_id",
                default_value="",
                description="Optional frame override for the recorded path.",
            ),
            DeclareLaunchArgument(
                "follow_path_action",
                default_value="follow_path",
                description="FollowPath action server name.",
            ),
            DeclareLaunchArgument(
                "navigate_to_pose_action",
                default_value="navigate_to_pose",
                description="NavigateToPose action server name for /goto_point commands.",
            ),
            DeclareLaunchArgument(
                "controller_id",
                default_value="FollowPath",
                description="Nav2 controller plugin id.",
            ),
            DeclareLaunchArgument(
                "goal_checker_id",
                default_value="",
                description="Optional Nav2 goal checker id.",
            ),
            DeclareLaunchArgument(
                "record_spacing",
                default_value="0.20",
                description="Distance in meters between recorded samples.",
            ),
            DeclareLaunchArgument(
                "final_append_distance",
                default_value="0.05",
                description="Append the latest pose on stop if it is farther than this threshold.",
            ),
            DeclareLaunchArgument(
                "return_mode",
                default_value="forward",
                description="Return mode: forward or reverse.",
            ),
            DeclareLaunchArgument(
                "final_orientation_mode",
                default_value="align_to_start",
                description="Forward-mode final orientation: align_to_start or match_path.",
            ),
            DeclareLaunchArgument(
                "auto_record_on_start",
                default_value="false",
                description="Start recording immediately after node startup.",
            ),
            DeclareLaunchArgument(
                "recorded_path_topic",
                default_value=default_recorded_topic,
                description="Preview topic for the recorded outbound path.",
            ),
            DeclareLaunchArgument(
                "return_path_topic",
                default_value=default_return_topic,
                description="Preview topic for the reconstructed return path.",
            ),
            DeclareLaunchArgument(
                "record_path_topic",
                default_value="/record_path",
                description="Bool topic: true starts path recording, false stops and saves it.",
            ),
            DeclareLaunchArgument(
                "repeat_path_topic",
                default_value="/repeat_path",
                description="Int8 topic: 1 forward repeat, -1 reverse repeat, 0 stop repeat.",
            ),
            DeclareLaunchArgument(
                "goto_point_topic",
                default_value="/goto_point",
                description="Point topic used to trigger NavigateToPose goals.",
            ),
            DeclareLaunchArgument(
                "status_topic",
                default_value="/Status",
                description="String status feedback topic for the upper computer.",
            ),
            DeclareLaunchArgument(
                "goto_frame_id",
                default_value="map",
                description="Frame used for /goto_point targets.",
            ),
            DeclareLaunchArgument(
                "goto_use_point_z_as_yaw",
                default_value="true",
                description="Interpret Point.z from /goto_point as target yaw in radians.",
            ),
            Node(
                package="fast_lio_sam_g",
                executable="track_return.py",
                name="track_return_manager",
                output="screen",
                parameters=[
                    {
                        "odom_topic": LaunchConfiguration("odom_topic"),
                        "frame_id": LaunchConfiguration("frame_id"),
                        "follow_path_action": LaunchConfiguration("follow_path_action"),
                        "navigate_to_pose_action": LaunchConfiguration("navigate_to_pose_action"),
                        "controller_id": LaunchConfiguration("controller_id"),
                        "goal_checker_id": LaunchConfiguration("goal_checker_id"),
                        "record_spacing": LaunchConfiguration("record_spacing"),
                        "final_append_distance": LaunchConfiguration("final_append_distance"),
                        "return_mode": LaunchConfiguration("return_mode"),
                        "final_orientation_mode": LaunchConfiguration("final_orientation_mode"),
                        "auto_record_on_start": LaunchConfiguration("auto_record_on_start"),
                        "recorded_path_topic": LaunchConfiguration("recorded_path_topic"),
                        "return_path_topic": LaunchConfiguration("return_path_topic"),
                        "record_path_topic": LaunchConfiguration("record_path_topic"),
                        "repeat_path_topic": LaunchConfiguration("repeat_path_topic"),
                        "goto_point_topic": LaunchConfiguration("goto_point_topic"),
                        "status_topic": LaunchConfiguration("status_topic"),
                        "goto_frame_id": LaunchConfiguration("goto_frame_id"),
                        "goto_use_point_z_as_yaw": LaunchConfiguration("goto_use_point_z_as_yaw"),
                    }
                ],
            ),
        ]
    )
