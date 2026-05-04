#!/usr/bin/env python3

import copy
import math
from functools import partial

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import FollowPath, NavigateToPose
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Int8, String
from std_srvs.srv import Trigger


def yaw_from_quaternion(orientation):
    siny_cosp = 2.0 * (
        orientation.w * orientation.z + orientation.x * orientation.y
    )
    cosy_cosp = 1.0 - 2.0 * (
        orientation.y * orientation.y + orientation.z * orientation.z
    )
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    half_yaw = 0.5 * yaw
    return (
        0.0,
        0.0,
        math.sin(half_yaw),
        math.cos(half_yaw),
    )


def distance_2d(first_pose, second_pose):
    dx = first_pose.pose.position.x - second_pose.pose.position.x
    dy = first_pose.pose.position.y - second_pose.pose.position.y
    return math.hypot(dx, dy)


class TrackReturnManager(Node):
    def __init__(self):
        super().__init__("track_return_manager")

        self.declare_parameter("odom_topic", "/OdometryHighFreq")
        self.declare_parameter("frame_id", "")
        self.declare_parameter("follow_path_action", "follow_path")
        self.declare_parameter("controller_id", "FollowPath")
        self.declare_parameter("goal_checker_id", "")
        self.declare_parameter("record_spacing", 0.20)
        self.declare_parameter("final_append_distance", 0.05)
        self.declare_parameter("return_mode", "forward")
        self.declare_parameter("final_orientation_mode", "align_to_start")
        self.declare_parameter("wait_for_action_server_sec", 2.0)
        self.declare_parameter("auto_record_on_start", False)
        self.declare_parameter("auto_record_on_goto", True)
        self.declare_parameter("recorded_path_topic", "~/recorded_path")
        self.declare_parameter("return_path_topic", "~/return_path")
        self.declare_parameter("record_path_topic", "/record_path")
        self.declare_parameter("repeat_path_topic", "/repeat_path")
        self.declare_parameter("goto_point_topic", "/goto_point")
        self.declare_parameter("status_topic", "/Status")
        self.declare_parameter("navigate_to_pose_action", "navigate_to_pose")
        self.declare_parameter("goto_frame_id", "map")
        self.declare_parameter("goto_use_point_z_as_yaw", True)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.frame_id_override = self.get_parameter("frame_id").value
        self.follow_path_action = self.get_parameter("follow_path_action").value
        self.controller_id = self.get_parameter("controller_id").value
        self.goal_checker_id = self.get_parameter("goal_checker_id").value
        self.record_spacing = max(0.01, float(self.get_parameter("record_spacing").value))
        self.final_append_distance = max(
            0.0, float(self.get_parameter("final_append_distance").value)
        )
        self.return_mode = str(self.get_parameter("return_mode").value).strip().lower()
        self.final_orientation_mode = str(
            self.get_parameter("final_orientation_mode").value
        ).strip().lower()
        self.wait_for_action_server_sec = max(
            0.1, float(self.get_parameter("wait_for_action_server_sec").value)
        )
        self.auto_record_on_start = bool(
            self.get_parameter("auto_record_on_start").value
        )
        self.auto_record_on_goto = bool(
            self.get_parameter("auto_record_on_goto").value
        )
        recorded_path_topic = self.get_parameter("recorded_path_topic").value
        return_path_topic = self.get_parameter("return_path_topic").value
        self.record_path_topic = str(self.get_parameter("record_path_topic").value)
        self.repeat_path_topic = str(self.get_parameter("repeat_path_topic").value)
        self.goto_point_topic = str(self.get_parameter("goto_point_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.navigate_to_pose_action = str(
            self.get_parameter("navigate_to_pose_action").value
        )
        self.goto_frame_id = str(self.get_parameter("goto_frame_id").value).strip()
        self.goto_use_point_z_as_yaw = bool(
            self.get_parameter("goto_use_point_z_as_yaw").value
        )

        if self.return_mode not in ("forward", "reverse"):
            raise ValueError("return_mode must be 'forward' or 'reverse'")
        if self.final_orientation_mode not in ("align_to_start", "match_path"):
            raise ValueError(
                "final_orientation_mode must be 'align_to_start' or 'match_path'"
            )

        self.is_recording = False
        self.latest_pose = None
        self.recorded_poses = []
        self.recorded_frame_id = ""
        self.goal_handle = None
        self.goal_pending = False
        self.goal_kind = ""
        self.goal_pending_kind = ""
        self.pending_goto_pose = None

        odom_qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        path_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            odom_qos,
        )
        self.recorded_path_publisher = self.create_publisher(
            Path, recorded_path_topic, path_qos
        )
        self.return_path_publisher = self.create_publisher(
            Path, return_path_topic, path_qos
        )
        self.status_publisher = self.create_publisher(String, self.status_topic, 10)

        self.follow_path_client = ActionClient(
            self, FollowPath, self.follow_path_action
        )
        self.navigate_to_pose_client = ActionClient(
            self, NavigateToPose, self.navigate_to_pose_action
        )

        self.create_subscription(Bool, self.record_path_topic, self.record_path_callback, 10)
        self.create_subscription(Int8, self.repeat_path_topic, self.repeat_path_callback, 10)
        self.create_subscription(Point, self.goto_point_topic, self.goto_point_callback, 10)

        self.create_service(Trigger, "~/start_recording", self.start_recording_callback)
        self.create_service(Trigger, "~/stop_recording", self.stop_recording_callback)
        self.create_service(Trigger, "~/clear_path", self.clear_path_callback)
        self.create_service(Trigger, "~/return_to_start", self.return_to_start_callback)
        self.create_service(Trigger, "~/cancel_return", self.cancel_return_callback)

        self.get_logger().info(
            f"track return ready: odom={self.odom_topic}, repeat_default={self.return_mode}, "
            f"follow_path={self.follow_path_action}"
        )
        self.get_logger().info(
            f"command topics: record={self.record_path_topic}, repeat={self.repeat_path_topic}, "
            f"goto={self.goto_point_topic}, status={self.status_topic}"
        )
        if self.return_mode == "reverse":
            self.get_logger().warn(
                "reverse mode requires Nav2 controller allow_reversing=true"
            )

        self.publish_recorded_path()
        self.publish_return_path(None)

        if self.auto_record_on_start:
            self.start_recording(reset_path=True)

    def odom_callback(self, msg):
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.frame_id_override or msg.header.frame_id or "odom"
        pose.pose = msg.pose.pose

        self.latest_pose = pose

        if not self.is_recording:
            return

        self.append_pose_if_needed(pose, min_distance=self.record_spacing)

    def start_recording(self, reset_path):
        if self.goal_pending or self.goal_handle is not None:
            return False, "A navigation action is already active."

        if reset_path:
            self.recorded_poses = []
            self.recorded_frame_id = ""

        self.is_recording = True

        if self.latest_pose is not None:
            self.append_pose_if_needed(self.latest_pose, min_distance=0.0, force=True)
            return True, f"Recording started with {len(self.recorded_poses)} seed pose(s)."

        return True, "Recording started. Waiting for odometry."

    def stop_recording(self):
        self.is_recording = False
        if self.latest_pose is not None and self.recorded_poses:
            self.append_pose_if_needed(
                self.latest_pose,
                min_distance=self.final_append_distance,
                force=True,
            )
        return len(self.recorded_poses)

    def clear_recorded_path(self):
        self.is_recording = False
        self.recorded_poses = []
        self.recorded_frame_id = ""
        self.publish_recorded_path()
        self.publish_return_path(None)

    def append_pose_if_needed(self, pose, min_distance, force=False):
        if not self.recorded_frame_id:
            self.recorded_frame_id = pose.header.frame_id
        elif pose.header.frame_id != self.recorded_frame_id:
            self.get_logger().warn(
                f"Ignoring odometry frame {pose.header.frame_id}, expected {self.recorded_frame_id}"
            )
            return False

        if not self.recorded_poses:
            self.recorded_poses.append(self.copy_pose(pose))
            self.publish_recorded_path()
            return True

        last_pose = self.recorded_poses[-1]
        distance = distance_2d(pose, last_pose)
        if force:
            if distance < max(1.0e-4, min_distance):
                return False
        elif distance < min_distance:
            return False

        self.recorded_poses.append(self.copy_pose(pose))
        self.publish_recorded_path()
        return True

    def build_repeat_path(self, path_direction=None):
        path_direction = (path_direction or self.return_mode).strip().lower()
        if path_direction not in ("forward", "reverse"):
            raise RuntimeError("path_direction must be 'forward' or 'reverse'")

        if len(self.recorded_poses) < 2:
            raise RuntimeError("Need at least 2 recorded poses to build a repeat path")

        original_path = [self.copy_pose(pose) for pose in self.recorded_poses]
        ordered_path = (
            original_path if path_direction == "forward" else list(reversed(original_path))
        )

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.recorded_frame_id or ordered_path[0].header.frame_id

        previous_yaw = yaw_from_quaternion(ordered_path[0].pose.orientation)
        final_source_yaw = yaw_from_quaternion(ordered_path[-1].pose.orientation)

        for index, source_pose in enumerate(ordered_path):
            pose = PoseStamped()
            pose.header.stamp = path.header.stamp
            pose.header.frame_id = path.header.frame_id
            pose.pose.position.x = source_pose.pose.position.x
            pose.pose.position.y = source_pose.pose.position.y
            pose.pose.position.z = source_pose.pose.position.z

            if index < len(ordered_path) - 1:
                next_pose = ordered_path[index + 1]
                dx = next_pose.pose.position.x - source_pose.pose.position.x
                dy = next_pose.pose.position.y - source_pose.pose.position.y
                yaw = math.atan2(dy, dx)
                previous_yaw = yaw
            else:
                if self.final_orientation_mode == "match_path":
                    yaw = final_source_yaw
                else:
                    yaw = previous_yaw

            qx, qy, qz, qw = quaternion_from_yaw(yaw)
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            path.poses.append(pose)

        return path

    def publish_recorded_path(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.recorded_frame_id or self.frame_id_override or "odom"
        path.poses = [self.copy_pose(pose) for pose in self.recorded_poses]
        for pose in path.poses:
            pose.header.stamp = path.header.stamp
            pose.header.frame_id = path.header.frame_id
        self.recorded_path_publisher.publish(path)

    def publish_return_path(self, path):
        if path is not None:
            self.return_path_publisher.publish(path)
            return

        empty_path = Path()
        empty_path.header.stamp = self.get_clock().now().to_msg()
        empty_path.header.frame_id = self.recorded_frame_id or self.frame_id_override or "odom"
        self.return_path_publisher.publish(empty_path)

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_publisher.publish(msg)

    def record_path_callback(self, msg):
        if msg.data:
            success, message = self.start_recording(reset_path=True)
            if success:
                self.publish_status("record_start")
                self.get_logger().info(message)
            else:
                self.publish_status("record_rejected")
                self.get_logger().warn(message)
            return

        count = self.stop_recording()
        self.publish_status("record_stop")
        self.publish_status("save")
        self.get_logger().info(f"Recording stopped from topic command with {count} pose(s)")

    def repeat_path_callback(self, msg):
        command = int(msg.data)
        if command == 0:
            if not self.goal_handle and self.goal_kind != "repeat" and self.goal_pending_kind != "repeat":
                self.publish_status("repeat_stop")
                self.get_logger().info("Repeat stop requested with no active repeat action")
                return

            success, message = self.cancel_navigation("repeat")
            if not success:
                self.get_logger().warn(message)
            return

        if command == 1:
            path_direction = "forward"
        elif command == -1:
            path_direction = "reverse"
        else:
            self.publish_status("repeat_invalid")
            self.get_logger().warn(f"Unsupported repeat_path command: {command}")
            return

        success, message = self.start_repeat(path_direction)
        if not success:
            self.publish_status("repeat_failed")
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    def goto_point_callback(self, msg):
        # 添加打印，显示接收到的点坐标
        self.get_logger().info(f"Received goto point: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}")

        goal_pose = self.build_goto_pose(msg)
        self.publish_status("goto_point_received")

        if self.goal_pending or self.goal_handle is not None:
            if self.goal_kind == "goto_point" or self.goal_pending_kind == "goto_point":
                self.pending_goto_pose = goal_pose
                self.publish_status("goto_point_update")
                self.get_logger().info(
                    "Received a new goto target while goto is active; replacing it"
                )

                if self.goal_handle is not None and self.goal_kind == "goto_point":
                    success, message = self.cancel_navigation("goto_point")
                    if not success:
                        self.get_logger().warn(message)
                return

            self.publish_status("goto_point_failed")
            self.get_logger().warn(
                "Received goto command while a non-goto navigation action is active"
            )
            return

        success, message = self.start_goto_point(goal_pose)
        if not success:
            self.publish_status("goto_point_failed")
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    def build_goto_pose(self, point_msg):
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = self.goto_frame_id or self.recorded_frame_id or "map"
        goal_pose.pose.position.x = float(point_msg.x)
        goal_pose.pose.position.y = float(point_msg.y)
        goal_pose.pose.position.z = 0.0 if self.goto_use_point_z_as_yaw else float(point_msg.z)

        if self.goto_use_point_z_as_yaw:
            yaw = float(point_msg.z)
        elif self.latest_pose is not None and self.latest_pose.header.frame_id == goal_pose.header.frame_id:
            dx = goal_pose.pose.position.x - self.latest_pose.pose.position.x
            dy = goal_pose.pose.position.y - self.latest_pose.pose.position.y
            if math.hypot(dx, dy) > 1.0e-4:
                yaw = math.atan2(dy, dx)
            else:
                yaw = yaw_from_quaternion(self.latest_pose.pose.orientation)
        else:
            yaw = 0.0

        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        goal_pose.pose.orientation.x = qx
        goal_pose.pose.orientation.y = qy
        goal_pose.pose.orientation.z = qz
        goal_pose.pose.orientation.w = qw
        return goal_pose

    def start_repeat(self, path_direction):
        if self.goal_pending or self.goal_handle is not None:
            return False, "A navigation action is already active."

        if self.is_recording:
            pose_count = self.stop_recording()
            self.publish_status("record_stop")
            self.publish_status("save")
            self.get_logger().info(
                f"Stopped recording before repeat with {pose_count} pose(s)"
            )

        try:
            repeat_path = self.build_repeat_path(path_direction)
        except RuntimeError as exc:
            return False, str(exc)

        self.publish_return_path(repeat_path)

        if not self.follow_path_client.wait_for_server(
            timeout_sec=self.wait_for_action_server_sec
        ):
            return False, f"FollowPath action '{self.follow_path_action}' is not ready."

        goal = FollowPath.Goal()
        goal.path = repeat_path
        goal.controller_id = self.controller_id
        goal.goal_checker_id = self.goal_checker_id

        self.goal_pending = True
        self.goal_pending_kind = "repeat"
        send_goal_future = self.follow_path_client.send_goal_async(
            goal,
            feedback_callback=self.follow_path_feedback_callback,
        )
        send_goal_future.add_done_callback(
            partial(self.navigation_goal_response_callback, "repeat")
        )

        return True, (
            f"Sent {path_direction} repeat path with {len(repeat_path.poses)} pose(s) in "
            f"{repeat_path.header.frame_id}."
        )

    def start_goto_point(self, goal_pose):
        if self.goal_pending or self.goal_handle is not None:
            return False, "A navigation action is already active."

        auto_started_recording = False
        if self.auto_record_on_goto and not self.is_recording:
            success, message = self.start_recording(reset_path=True)
            if not success:
                return False, message
            auto_started_recording = True
            self.publish_status("record_start")
            self.get_logger().info(
                f"Auto-started recording for goto command: {message}"
            )

        if not self.navigate_to_pose_client.wait_for_server(
            timeout_sec=self.wait_for_action_server_sec
        ):
            if auto_started_recording:
                self.clear_recorded_path()
                self.publish_status("record_discarded")
            return False, (
                f"NavigateToPose action '{self.navigate_to_pose_action}' is not ready."
            )

        goal = NavigateToPose.Goal()
        goal.pose = goal_pose

        self.goal_pending = True
        self.goal_pending_kind = "goto_point"
        send_goal_future = self.navigate_to_pose_client.send_goal_async(goal)
        send_goal_future.add_done_callback(
            partial(self.navigation_goal_response_callback, "goto_point")
        )

        return True, (
            f"Sent goto goal to ({goal_pose.pose.position.x:.3f}, "
            f"{goal_pose.pose.position.y:.3f}) in {goal_pose.header.frame_id}."
        )

    def cancel_navigation(self, expected_kind):
        if self.goal_handle is None:
            return False, f"No active {expected_kind} action to cancel."
        if self.goal_kind != expected_kind:
            return False, f"Active navigation action is '{self.goal_kind}', not '{expected_kind}'."

        cancel_future = self.goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(
            partial(self.navigation_cancel_done_callback, expected_kind)
        )
        return True, f"Cancel requested for active {expected_kind} action."

    def navigation_goal_response_callback(self, kind, future):
        self.goal_pending = False
        self.goal_pending_kind = ""
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to send {kind} goal: {exc}")
            if kind == "repeat":
                self.publish_status("repeat_failed")
            else:
                if self.is_recording:
                    self.clear_recorded_path()
                    self.publish_status("record_discarded")
                self.publish_status("goto_point_failed")
                self.start_pending_goto_if_any()
            return

        if not goal_handle.accepted:
            self.get_logger().warn(f"{kind} goal was rejected")
            if kind == "repeat":
                self.publish_status("repeat_failed")
            else:
                if self.is_recording:
                    self.clear_recorded_path()
                    self.publish_status("record_discarded")
                self.publish_status("goto_point_failed")
                self.start_pending_goto_if_any()
            return

        self.goal_handle = goal_handle
        self.goal_kind = kind

        # ========== 关键修改1 ==========
        # 对于 goto_point，不发送 goto_point_start 和 goto_point，避免上层误以为到达
        if kind == "repeat":
            self.publish_status("repeat_start")
        else:
            # 原代码会发布 goto_point_start 和 goto_point，现在注释掉
            self.get_logger().debug("goto_point goal accepted (no status published to avoid false arrival)")
        # ==============================

        self.get_logger().info(f"{kind} goal accepted")

        if kind == "goto_point" and self.pending_goto_pose is not None:
            success, message = self.cancel_navigation("goto_point")
            if not success:
                self.get_logger().warn(message)

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(partial(self.navigation_result_callback, kind))

    def navigation_result_callback(self, kind, future):
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to get {kind} result: {exc}")
            if kind == "repeat":
                self.publish_status("repeat_failed")
            else:
                self.publish_status("goto_point_failed")
            self.goal_handle = None
            self.goal_kind = ""
            self.start_pending_goto_if_any()
            return

        status = result.status
        if kind == "repeat":
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("Repeat path execution succeeded")
                self.publish_status("repeat_stop")
            elif status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warn("Repeat path execution canceled")
                self.publish_status("repeat_stop")
            else:
                self.get_logger().warn(f"Repeat path execution finished with status {status}")
                self.publish_status("repeat_failed")
        else:
            # ========== 关键修改2 ==========
            # 真正到达时发布 "goto_point" 而不是 "goto_point_reached"
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("Goto point execution succeeded")
                self.publish_status("goto_point")   # 原为 goto_point_reached
                if self.is_recording:
                    pose_count = self.stop_recording()
                    self.publish_status("record_stop")
                    self.publish_status("save")
                    self.get_logger().info(
                        f"Goto completed and saved recorded path with {pose_count} pose(s)"
                    )
            elif status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warn("Goto point execution canceled")
                self.publish_status("goto_point_canceled")
                if self.is_recording:
                    self.clear_recorded_path()
                    self.publish_status("record_discarded")
                    self.get_logger().warn(
                        "Goto canceled; cleared the partially recorded path"
                    )
            else:
                self.get_logger().warn(f"Goto point execution finished with status {status}")
                self.publish_status("goto_point_failed")
                if self.is_recording:
                    self.clear_recorded_path()
                    self.publish_status("record_discarded")
                    self.get_logger().warn(
                        "Goto did not succeed; cleared the partially recorded path"
                    )
            # ==============================

        self.goal_handle = None
        self.goal_kind = ""
        self.start_pending_goto_if_any()

    def navigation_cancel_done_callback(self, kind, future):
        try:
            cancel_response = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to cancel {kind} goal: {exc}")
            return

        if cancel_response.goals_canceling:
            self.get_logger().info(f"{kind} cancel accepted")
        else:
            self.get_logger().warn(f"{kind} cancel request was not accepted")

    def start_pending_goto_if_any(self):
        if self.pending_goto_pose is None:
            return
        if self.goal_pending or self.goal_handle is not None:
            return

        goal_pose = self.pending_goto_pose
        self.pending_goto_pose = None
        success, message = self.start_goto_point(goal_pose)
        if not success:
            self.publish_status("goto_point_failed")
            self.get_logger().warn(f"Failed to start queued goto target: {message}")
            return

        self.get_logger().info(f"Queued goto target started: {message}")

    def start_recording_callback(self, request, response):
        del request
        success, message = self.start_recording(reset_path=True)
        response.success = success
        response.message = message
        return response

    def stop_recording_callback(self, request, response):
        del request
        count = self.stop_recording()
        response.success = True
        response.message = f"Recording stopped with {count} pose(s)."
        return response

    def clear_path_callback(self, request, response):
        del request
        if self.goal_pending or self.goal_handle is not None:
            response.success = False
            response.message = "Cannot clear path while a navigation action is active."
            return response

        self.clear_recorded_path()
        response.success = True
        response.message = "Recorded path cleared."
        return response

    def return_to_start_callback(self, request, response):
        del request
        success, message = self.start_repeat("reverse")
        response.success = True
        response.success = success
        response.message = message
        return response

    def cancel_return_callback(self, request, response):
        del request
        success, message = self.cancel_navigation("repeat")
        response.success = success
        response.message = message
        return response

    def follow_path_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().debug(
            f"distance_to_goal={feedback.distance_to_goal:.3f}, speed={feedback.speed:.3f}"
        )

    @staticmethod
    def copy_pose(source_pose):
        return copy.deepcopy(source_pose)


def main(args=None):
    rclpy.init(args=args)
    node = TrackReturnManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()