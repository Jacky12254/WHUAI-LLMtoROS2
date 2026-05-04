#!/usr/bin/env python3
"""
person_camera_tracker.py
纯相机检测与跟踪模块，不依赖ROS2。
负责：
  - RealSense 相机初始化与图像获取
  - YOLO 行人检测
  - DeepSort 多目标跟踪
  - 3D 坐标解算
  - OpenCV 可视化显示

可独立运行（纯视觉跟踪），也可被 ROS2 节点导入使用。
"""

import cv2
import time
import math
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# 导入 DeepSort（位于 deep_sort 子模块中）
from .deep_sort import DeepSort


class PersonCameraTracker:
    """
    不依赖 ROS2 的相机 + 检测 + 跟踪模块。
    封装了 RealSense 相机操作、YOLO 检测、DeepSort 跟踪、3D 坐标解算和可视化。
    """

    def __init__(self,
                 model_path='yolov8n.pt',
                 deepsort_checkpoint='deep_sort/deep_sort/deep/checkpoint/ckpt.t7',
                 detect_class=0,
                 save_point_cloud=True,
                 point_cloud_path="point_cloud_data.txt"):
        """
        初始化相机、检测模型和跟踪器。

        参数:
            model_path: YOLO 模型权重路径
            deepsort_checkpoint: DeepSort 特征提取器权重路径
            detect_class: 检测目标类别（0=person）
            save_point_cloud: 是否保存点云数据到文件
            point_cloud_path: 点云数据保存路径
        """
        # --- 加载模型 ---
        self.model = YOLO(model_path)
        self.tracker = DeepSort(deepsort_checkpoint)
        self.detect_class = detect_class
        print(f"[CameraTracker] 检测目标: {self.model.names[self.detect_class]}")

        # --- 点云保存 ---
        self.save_point_cloud = save_point_cloud
        self.point_cloud_path = point_cloud_path

        # --- RealSense 相机配置 ---
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # --- FPS 计算 ---
        self.frame_count = 0
        self.start_time = time.time()

        print("[CameraTracker] 相机启动完成")

    def get_aligned_images(self):
        """
        获取对齐后的彩色图和深度图。

        返回:
            intr: 彩色相机内参
            depth_intrin: 深度相机内参
            color_image: 彩色图像 (numpy.ndarray, HxWx3, BGR)
            depth_image: 深度图像 (numpy.ndarray, HxW, uint16)
            aligned_depth_frame: 对齐后的深度帧（用于 get_distance）
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return intr, depth_intrin, color_image, depth_image, aligned_depth_frame

    def get_3d_camera_coordinate(self, depth_pixel, aligned_depth_frame, depth_intrin):
        """
        获取像素点的3D相机坐标（相对于相机坐标系）。

        参数:
            depth_pixel: [x, y] 像素坐标
            aligned_depth_frame: 对齐后的深度帧
            depth_intrin: 深度相机内参

        返回:
            dis: 深度值（米）
            camera_coordinate: [x, y, z] 3D坐标（米）
        """
        x, y = depth_pixel[0], depth_pixel[1]
        dis = aligned_depth_frame.get_distance(x, y)
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
        return dis, camera_coordinate

    @staticmethod
    def extract_detections(results, detect_class):
        """
        从 YOLO 结果中提取指定类别的检测信息。

        参数:
            results: YOLO 推理结果
            detect_class: 目标类别ID

        返回:
            detections: (N, 4) 数组，每行为 [x_center, y_center, width, height]
            confarray: 置信度列表
        """
        detections = np.empty((0, 4))
        confarray = []

        for r in results:
            for box in r.boxes:
                if box.cls[0].int() == detect_class:
                    x1, y1, x2, y2 = box.xywh[0].int().tolist()
                    conf = round(box.conf[0].item(), 2)
                    detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                    confarray.append(conf)
        return detections, confarray

    @staticmethod
    def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=1, text_color=(255, 255, 255),
                              bg_color=(0, 0, 0), thickness=1):
        """绘制带有背景框的文本（不依赖ROS2）。"""
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        bottom_left = origin
        top_right = (origin[0] + text_width, origin[1] - text_height - 5)
        cv2.rectangle(img, bottom_left, top_right, bg_color, -1)
        text_origin = (origin[0], origin[1] - 5)
        cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    def save_point_cloud_data(self, camera_coordinate, track_id):
        """保存点云数据到文件。"""
        if not self.save_point_cloud:
            return
        try:
            with open(self.point_cloud_path, "a") as file:
                file.write(f"\nTime: {time.time()}, TrackID: {track_id}, "
                           f"Coordinate: {camera_coordinate}\n")
        except Exception as e:
            print(f"[CameraTracker] 保存点云数据失败: {e}")

    def process_frame(self):
        """
        处理一帧图像：获取图像 → YOLO检测 → DeepSort跟踪 → 3D坐标解算。

        返回一个字典，包含所有跟踪结果（纯 Python 数据结构，无 ROS2 依赖）：

        {
            "color_image": numpy.ndarray,       # 标注后的彩色图
            "depth_image": numpy.ndarray,        # 深度图
            "detections": [                      # 检测/跟踪结果列表
                {
                    "track_id": int,             # 跟踪ID
                    "bbox": [x1, y1, x2, y2],    # 边界框像素坐标
                    "center": [ux, uy],           # 中心点像素坐标
                    "depth": float,              # 深度值（米）
                    "camera_coordinate": [x, y, z],  # 3D相机坐标（米）
                },
                ...
            ],
            "fps": float,                        # 当前FPS
            "intr": rs.intrinsics,              # 彩色相机内参
            "depth_intrin": rs.intrinsics,       # 深度相机内参
            "aligned_depth_frame": rs.depth_frame,  # 对齐深度帧
        }

        如果获取图像失败，返回 None。
        """
        # --- 获取对齐图像 ---
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame = \
            self.get_aligned_images()

        if not depth_image.any() or not color_image.any():
            return None

        # --- YOLO 检测 ---
        results = self.model(color_image, stream=True)
        detections, confarray = self.extract_detections(results, self.detect_class)

        # --- DeepSort 跟踪 ---
        results_tracker = self.tracker.update(detections, confarray, color_image)

        # --- 构建结果列表 ---
        tracking_results = []
        for item in results_tracker:
            x1, y1, x2, y2 = map(int, [item[0], item[1], item[2], item[3]])
            track_id = int(item[4])

            # 计算中心点
            ux = int((x1 + x2) / 2)
            uy = int((y1 + y2) / 2)

            # 获取3D坐标
            dis, camera_coordinate = self.get_3d_camera_coordinate(
                [ux, uy], aligned_depth_frame, depth_intrin
            )

            tracking_results.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "center": [ux, uy],
                "depth": dis,
                "camera_coordinate": [
                    round(camera_coordinate[0], 4),
                    round(camera_coordinate[1], 4),
                    round(camera_coordinate[2], 4),
                ],
            })

            # 可选：保存点云
            self.save_point_cloud_data(camera_coordinate, track_id)

        # --- 计算 FPS ---
        self.frame_count += 1
        current_time = time.time()
        fps = self.frame_count / (current_time - self.start_time)

        return {
            "color_image": color_image,
            "depth_image": depth_image,
            "detections": tracking_results,
            "fps": round(fps, 2),
            "intr": intr,
            "depth_intrin": depth_intrin,
            "aligned_depth_frame": aligned_depth_frame,
        }

    def draw_detections(self, frame_data):
        """
        在图像上绘制检测/跟踪结果（边界框、ID、3D坐标、FPS）。

        参数:
            frame_data: process_frame() 返回的帧数据字典
        """
        if frame_data is None:
            return

        color_image = frame_data["color_image"]
        fps = frame_data["fps"]

        for det in frame_data["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            ux, uy = det["center"]
            track_id = det["track_id"]
            cam_coord = det["camera_coordinate"]

            # 绘制边界框
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # 绘制跟踪ID（带背景）
            self.putTextWithBackground(
                color_image, str(track_id),
                (max(-10, x1), max(40, y1)),
                font_scale=1.5, text_color=(255, 255, 255),
                bg_color=(255, 0, 255)
            )

            # 绘制中心点
            cv2.circle(color_image, (ux, uy), 4, (255, 255, 255), 5)

            # 绘制3D坐标
            formatted_coord = f"({cam_coord[0]:.2f}, {cam_coord[1]:.2f}, {cam_coord[2]:.2f})"
            cv2.putText(color_image, formatted_coord, (ux + 20, uy + 10),
                        0, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # 绘制FPS
        cv2.putText(color_image, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def run_standalone(self):
        """
        独立运行模式（无ROS2）。
        持续从相机读取帧、检测跟踪、显示结果。
        按 'q' 或 ESC 退出。
        """
        print("[CameraTracker] 进入独立运行模式，按 'q' 或 ESC 退出...")

        try:
            while True:
                frame_data = self.process_frame()
                if frame_data is None:
                    continue

                # 绘制结果
                self.draw_detections(frame_data)

                # 显示图像
                cv2.imshow('Person Tracking (Standalone)', frame_data["color_image"])
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 或 ESC
                    break

        except KeyboardInterrupt:
            print("[CameraTracker] 用户中断")
        finally:
            self.cleanup()

    def cleanup(self):
        """释放相机资源和关闭窗口。"""
        print("[CameraTracker] 清理资源...")
        self.pipeline.stop()
        cv2.destroyAllWindows()


# ============================
# 独立运行入口
# ============================
def main():
    """
    独立运行入口。
    用法: python -m person_tracking_ros.person_camera_tracker
    """
    import argparse

    parser = argparse.ArgumentParser(description="Person Camera Tracker (Standalone)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO模型路径")
    parser.add_argument("--checkpoint", type=str,
                        default="deep_sort/deep_sort/deep/checkpoint/ckpt.t7",
                        help="DeepSort checkpoint路径")
    parser.add_argument("--class_id", type=int, default=0,
                        help="检测类别ID (0=person)")
    parser.add_argument("--no_pointcloud", action="store_true",
                        help="禁用点云保存")
    parser.add_argument("--pointcloud_path", type=str,
                        default="point_cloud_data.txt",
                        help="点云数据保存路径")
    args = parser.parse_args()

    tracker = PersonCameraTracker(
        model_path=args.model,
        deepsort_checkpoint=args.checkpoint,
        detect_class=args.class_id,
        save_point_cloud=not args.no_pointcloud,
        point_cloud_path=args.pointcloud_path,
    )

    tracker.run_standalone()


if __name__ == "__main__":
    main()
