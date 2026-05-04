from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

from rm_ros_interfaces.msg import Jointposeeuler  # pyright: ignore
from yolo_realsense.transform_ros import camera_to_base

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover
    get_package_share_directory = None


class Detect4Node(Node):

    def __init__(self):
        super().__init__('detect4')

        self.declare_parameter('detection_topic', '/detection_result')
        self.declare_parameter('grab_status_topic', '/grab4_status')
        self.declare_parameter('pose_topic', '/rm_driver/udp_joint_pose_euler')
        self.declare_parameter('model_path', '')
        self.declare_parameter('mask_path', '')
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('image_size', 1280)
        self.declare_parameter('frame_period_sec', 0.05)

        detection_topic = self.get_parameter('detection_topic').value
        grab_status_topic = self.get_parameter('grab_status_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        frame_period_sec = float(self.get_parameter('frame_period_sec').value)

        self.publisher_ = self.create_publisher(String, detection_topic, 10)
        self.create_subscription(Jointposeeuler, pose_topic, self.pose_callback, 10)
        self.create_subscription(String, grab_status_topic, self.grab_status_callback, 10)
        self.process_timer = self.create_timer(frame_period_sec, self.process_frame)

        self.ee_pose = None
        self.model = None
        self.mask = None
        self.pipeline = None
        self.align = None
        self.session_active = False
        self.last_stop_reason = 'idle'
        self._waiting_pose_logged = False

        self.get_logger().info(
            'detect4 节点已启动，等待 /grab4_status 的 started 信号后开启识别'
        )

    def pose_callback(self, msg):
        self.ee_pose = [
            msg.position[0],
            msg.position[1],
            msg.position[2],
            msg.euler[0],
            msg.euler[1],
            msg.euler[2],
        ]
        if self.ee_pose is not None:
            self._waiting_pose_logged = False

    def grab_status_callback(self, msg: String):
        status = msg.data.strip()
        if not status:
            return

        self.get_logger().info(f'收到 grab4 状态: {status}')

        if status == 'started':
            self.start_session()
            return

        if status == 'success' or status == 'stopped' or status.startswith('failed:'):
            self.stop_session(status)

    def start_session(self):
        if self.session_active:
            self.get_logger().info('detect4 当前已在运行，忽略重复 started')
            return

        try:
            self.model = YOLO(str(self.resolve_asset_path('model_path', 'best.pt')))

            mask_path = self.resolve_asset_path('mask_path', 'mask.png')
            self.mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if self.mask is None:
                raise FileNotFoundError(f'无法加载 mask 文件: {mask_path}')
            if self.mask.shape != (720, 1280):
                raise ValueError(f'mask 尺寸必须是 (720, 1280)，当前是 {self.mask.shape}')

            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)

            self.session_active = True
            self.last_stop_reason = 'running'
            self._waiting_pose_logged = False
            self.get_logger().info('detect4 已开启，本轮抓取期间开始发布识别结果')
        except Exception as exc:
            self.get_logger().error(f'detect4 启动失败: {exc}')
            self.stop_session(f'failed_to_start:{exc}')

    def stop_session(self, reason='stopped'):
        was_active = self.session_active
        self.session_active = False
        self.last_stop_reason = reason

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as exc:
                self.get_logger().warn(f'关闭 RealSense 失败: {exc}')
            finally:
                self.pipeline = None

        self.align = None
        self.model = None
        self.mask = None
        cv2.destroyAllWindows()

        if was_active:
            self.get_logger().info(f'detect4 已关闭，原因: {reason}')

    def resolve_asset_path(self, parameter_name, default_name):
        configured_path = str(self.get_parameter(parameter_name).value).strip()
        if configured_path:
            candidate = Path(configured_path)
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(f'参数 {parameter_name} 指定的文件不存在: {candidate}')

        candidates = [Path(__file__).resolve().parent / default_name]

        if get_package_share_directory is not None:
            try:
                share_dir = Path(get_package_share_directory('yolo_realsense'))
                candidates.append(share_dir / 'assets' / default_name)
            except Exception:
                pass

        candidates.append(
            Path('/home/robocuphome/vision_ws/src/yolo_realsense/yolo_realsense') / default_name
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f'未找到资源文件 {default_name}，已检查: {", ".join(str(path) for path in candidates)}'
        )

    def get_aligned_images(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError('未获取到完整的深度/彩色图像帧')

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        return color_image, depth_image, depth_frame, depth_intrin

    def pixel_to_camera(self, pixel, depth_frame, depth_intrin):
        x, y = pixel
        distance = depth_frame.get_distance(x, y)
        return rs.rs2_deproject_pixel_to_point(depth_intrin, pixel, distance)

    def process_frame(self):
        if not self.session_active:
            return

        if self.ee_pose is None:
            if not self._waiting_pose_logged:
                self.get_logger().warn('detect4 等待机械臂位姿后再开始识别')
                self._waiting_pose_logged = True
            return

        try:
            color_image, depth_image, depth_frame, depth_intrin = self.get_aligned_images()
            if not depth_image.any() or not color_image.any():
                return

            masked_color_image = cv2.bitwise_and(color_image, color_image, mask=self.mask)
            results = self.model.predict(
                masked_color_image,
                conf=float(self.get_parameter('confidence').value),
                imgsz=int(self.get_parameter('image_size').value),
                verbose=False,
            )
            annotated_frame = results[0].plot()

            boxes = results[0].boxes.xyxy
            classes = results[0].boxes.cls
            names = results[0].names

            msg_text = ''
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cam_coord = self.pixel_to_camera([cx, cy], depth_frame, depth_intrin)
                base_coord = camera_to_base(cam_coord, self.ee_pose)

                label = names[int(classes[i])]
                msg_text += (
                    f'{label}:{base_coord[0]:.3f},{base_coord[1]:.3f},{base_coord[2]:.3f};'
                )

                cv2.circle(annotated_frame, (cx, cy), 4, (255, 255, 255), 5)
                cv2.putText(
                    annotated_frame,
                    f'{base_coord[0]:.2f},{base_coord[1]:.2f},{base_coord[2]:.2f}',
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            if msg_text:
                msg = String()
                msg.data = msg_text
                self.publisher_.publish(msg)

            display_img = cv2.resize(annotated_frame, (848, 480))
            cv2.imshow('Detect4', display_img)
            cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().error(f'detect4 运行异常，停止本轮识别: {exc}')
            self.stop_session(f'error:{exc}')

    def destroy_node(self):
        self.stop_session('shutdown')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Detect4Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
