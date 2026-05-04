import os
import math
import time
import threading
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import sys
# 将deepsort的路径添加到系统路径中，确保可以导入
sys.path.append(os.path.join(os.path.dirname(__file__), "deep_sort/deep_sort/deep"))
from .deep_sort.deep_sort import deep_sort as ds # 确保你的 deep_sort 路径正确

class AsyncVisionTracker:
    def __init__(self, yolo_model="yolov8n.pt", deepsort_ckpt="track/yolov8_deepsort_tracking/deep_sort/deep_sort/deep/checkpoint/ckpt.t7", align_threshold=50):
        """
        异步视觉追踪器初始化
        """
        self.align_threshold = align_threshold
        self.center_line_x = 848 // 2
        
        # 1. 初始化模型
        self.yolo_model = YOLO(yolo_model)
        self.detect_class = 0  # person
        self.tracker = ds.DeepSort(deepsort_ckpt)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.face_detector = cv2.FaceDetectorYN_create(
            os.path.join(script_dir, "model/face_detection_yunet_2023mar.onnx"), "", (848, 480), 0.9, 0.3, 5000
        )
        
        # 2. 初始化 RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        # 3. 线程共享数据与锁
        self.lock = threading.Lock()
        self.is_running = True
        self.target_locked = False
        self.target_id = None
        
        # 这是外部读取的最新数据容器
        self.latest_data = {
            "locked": False,       # 是否已经锁定目标
            "face_offset": 0,      # 人脸与中心线差值
            "face_depth": 0.0,
            "body_offset": 0,      # 人体与中心线差值
            "body_depth": 0.0,
        }
        
        # 4. 启动后台处理线程
        self.vision_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.vision_thread.start()
        print("视觉后台线程已启动，正在全速运行...")

    def _get_distance_safe(self, depth_frame, x, y):
        h, w = depth_frame.get_height(), depth_frame.get_width()
        if 0 <= x < w and 0 <= y < h:
            return depth_frame.get_distance(int(x), int(y))
        return 0.0

    def _tracking_loop(self):
        """后台高频视觉处理循环，跑满 30FPS"""
        while self.is_running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]
            self.face_detector.setInputSize((w, h))

            # 准备本帧的临时数据
            frame_data = {"locked": self.target_locked, "face_offset": -9999, "face_depth": 0.0, "body_offset": -9999, "body_depth": 0.0}

            # 1. 识别人脸
            _, faces = self.face_detector.detect(color_image)
            target_face = None
            if faces is not None and len(faces) > 0:
                # 找最大人脸
                target_face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = target_face[0:4]
                fc_x = int(fx + fw / 2)
                fc_y = int(fy + fh / 2)
                
                frame_data["face_offset"] = fc_x - self.center_line_x
                frame_data["face_depth"] = self._get_distance_safe(depth_frame, fc_x, fc_y)

            # 2. 识别并追踪人体 (YOLO + DeepSORT)
            results = self.yolo_model(color_image, verbose=False)
            detections = np.empty((0, 4)) 
            confarray = []
            
            # 提取 YOLO 结果
            for r in results:
                for box in r.boxes:
                    if box.cls[0].int() == self.detect_class:
                        x1, y1, x2, y2 = box.xywh[0].int().tolist()
                        conf = round(box.conf[0].item(), 2)
                        detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                        confarray.append(conf)

            # DeepSORT 更新
            resultsTracker = self.tracker.update(detections, confarray, color_image)

            # 3. 锁定目标逻辑
            if not self.target_locked and target_face is not None:
                # 如果还没锁定，寻找包含最大人脸的 DeepSORT ID
                fx, fy, fw, fh = target_face[0:4]
                fc_x = int(fx + fw / 2)
                fc_y = int(fy + fh / 2)
                
                for x1, y1, x2, y2, Id in resultsTracker:
                    if x1 <= fc_x <= x2 and y1 <= fc_y <= y2:
                        self.target_id = Id
                        self.target_locked = True
                        frame_data["locked"] = True
                        print(f"[{time.strftime('%H:%M:%S')}] 目标已锁定! DeepSORT ID: {self.target_id}")
                        break

            # 4. 获取已锁定目标的身体数据
            if self.target_locked:
                target_found_in_frame = False
                for x1, y1, x2, y2, Id in resultsTracker:
                    if Id == self.target_id:
                        target_found_in_frame = True
                        bc_x = int((x1 + x2) / 2)
                        bc_y = int((y1 + y2) / 2)
                        
                        frame_data["body_offset"] = bc_x - self.center_line_x
                        frame_data["body_depth"] = self._get_distance_safe(depth_frame, bc_x, bc_y)
                        break
                
                # 可选：如果好几帧都没找到 target_id，可以重置锁定状态
                # if not target_found_in_frame:
                #     self.target_locked = False 
            
            # === 安全地将本帧数据写入共享变量 ===
            with self.lock:
                self.latest_data = frame_data
                
            # （如果需要调试，可以在这里加上 cv2.imshow，但强烈建议在正式给底盘发指令时去掉 UI 渲染以节省算力）

    def get_control_data(self):
        """
        外部调用的 API，无论你以 2Hz 还是 100Hz 调用，都会瞬间返回当前最新的偏航数据。
        绝对不会阻塞外部程序的循环。
        """
        with self.lock:
            # 拷贝一份字典返回，避免外部使用时数据被后台线程覆盖
            return self.latest_data.copy()

    def release(self):
        """安全停止线程和相机"""
        self.is_running = False
        self.vision_thread.join(timeout=2.0)
        self.pipeline.stop()
        print("视觉模块已安全关闭。")


# ================= 外部（如底盘控制节点）调用示例 =================
if __name__ == "__main__":
    
    # 1. 实例化追踪器（后台线程自动跑起来）
    tracker = AsyncVisionTracker()
    
    try:
        print("外部控制程序开始以 5Hz 频率获取数据...")
        
        while True:
            start_time = time.time()
            
            # 2. 极速读取最新状态！不卡顿！
            data = tracker.get_control_data()
            
            if data["locked"]:
                print(f"底盘控制 | 人脸偏移: {data['face_offset']:>4} (深:{data['face_depth']:.2f}m) | 人体偏移: {data['body_offset']:>4} (深:{data['body_depth']:.2f}m)")
                
                # TODO: 在这里编写你的底盘 PID 或差速控制逻辑
                # if data['body_offset'] > 50: turn_right()
                # elif data['body_offset'] < -50: turn_left()
                
            else:
                print("底盘控制 | 寻找目标中...")
                
            # 控制读取频率为 5Hz (0.2s)
            time.sleep(max(0, 0.2 - (time.time() - start_time)))
            
    except KeyboardInterrupt:
        tracker.release()