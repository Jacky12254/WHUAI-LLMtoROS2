from pathlib import Path

import rclpy
from rclpy.node import Node

import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO

from std_msgs.msg import String
from rm_ros_interfaces.msg import Jointposeeuler  # pyright: ignore

# 坐标转换
from yolo_realsense.transform_ros import camera_to_base


class DetectROS(Node):

    def __init__(self):
        super().__init__('detect_ros')

        # 发布检测结果
        self.publisher_ = self.create_publisher(String, 'detection_result', 10)

        # 末端位姿
        self.ee_pose = None

        # 订阅欧拉角话题
        self.create_subscription(
            Jointposeeuler,
            '/rm_driver/udp_joint_pose_euler',
            self.pose_callback,
            10
        )

        # YOLO模型
        # self.model = YOLO(
        #     "/home/robocuphome/vision/yolo/ultralytics-main/runs/train/exp6/weights/best.pt"
        # )
        self.model = YOLO(
            "/home/jacky/vision_ws/src/yolo_realsense/yolo_realsense/best.pt"
        )

        mask_path = Path('/home/jacky/vision_ws/src/yolo_realsense/yolo_realsense/mask.png')
        self.mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise FileNotFoundError(f"无法加载 mask 文件: {mask_path}")
        if self.mask.shape != (720, 1280):
            raise ValueError(f"mask 尺寸必须是 (720, 1280)，当前是 {self.mask.shape}")

        # RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        self.get_logger().info("DetectROS node started")

    # ===============================
    def pose_callback(self, msg):
        self.ee_pose = [
            msg.position[0],
            msg.position[1],
            msg.position[2],
            msg.euler[0],
            msg.euler[1],
            msg.euler[2]
        ]

    # ===============================
    def get_aligned_images(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        return color_image, depth_image, depth_frame, depth_intrin

    # ===============================
    def pixel_to_camera(self, pixel, depth_frame, depth_intrin):
        x, y = pixel
        distance = depth_frame.get_distance(x, y)

        camera_coordinate = rs.rs2_deproject_pixel_to_point(
            depth_intrin, pixel, distance
        )
        return camera_coordinate

    # ===============================
    def process_frame(self):
        if self.ee_pose is None:
            self.get_logger().warn("等待机械臂位姿...")
            return

        color_image, depth_image, depth_frame, depth_intrin = self.get_aligned_images()

        if not depth_image.any() or not color_image.any():
            return

        masked_color_image = cv2.bitwise_and(color_image, color_image, mask=self.mask)

        results = self.model.predict(masked_color_image, conf=0.5, imgsz=1280)
        annotated_frame = results[0].plot()

        boxes = results[0].boxes.xyxy
        classes = results[0].boxes.cls
        names = results[0].names

        msg_text = ""

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # 相机坐标
            cam_coord = self.pixel_to_camera([cx, cy], depth_frame, depth_intrin)

            # 转 base 坐标
            base_coord = camera_to_base(cam_coord, self.ee_pose)

            label = names[int(classes[i])]
            msg_text += f"{label}:{base_coord[0]:.3f},{base_coord[1]:.3f},{base_coord[2]:.3f};"

            # 可视化
            cv2.circle(annotated_frame, (cx, cy), 4, (255, 255, 255), 5)
            cv2.putText(
                annotated_frame,
                f"{base_coord[0]:.2f},{base_coord[1]:.2f},{base_coord[2]:.2f}",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        if msg_text:
            msg = String()
            msg.data = msg_text
            self.publisher_.publish(msg)

        display_img = cv2.resize(annotated_frame, (848, 480))
        cv2.imshow("DetectROS", display_img)
        cv2.waitKey(1)

    # ===============================
    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectROS()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.process_frame()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
