import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from .deep_sort import DeepSort
import requests
import time
import threading

class PersonTrackingServer:
    def __init__(self, ros2_bridge_url, model_path, deepsort_checkpoint, detect_class=0, save_point_cloud=False, local_display=False):
        self.ros2_bridge_url = ros2_bridge_url
        self.model = YOLO(model_path)
        self.tracker = DeepSort(deepsort_checkpoint)
        self.detect_class = detect_class
        self.local_display = local_display
        self.is_running_flag = False
        self.target_track_id = -1
        self.last_http_time = 0

    def is_running(self):
        return self.is_running_flag

    def set_target(self, track_id):
        self.target_track_id = track_id

    def stop(self):
        self.is_running_flag = False

    def get_3d_camera_coordinate(self, depth_pixel, aligned_depth_frame, depth_intrin):
        x, y = depth_pixel
        dis = aligned_depth_frame.get_distance(x, y)
        if dis == 0:
            return 0, [0, 0, 0]
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
        return dis, camera_coordinate

    def start(self, stop_event, llm_callback, target_track_id=-1):
        """核心线程逻辑，受 stop_event 控制"""
        self.target_track_id = target_track_id
        self.is_running_flag = True
        
        # 1. 只有在任务真正开始时，才抢占摄像头
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            pipeline.start(config)
            align = rs.align(rs.stream.color)
            print("[Tracker] 相机启动成功，开始跟踪...")
        except Exception as e:
            print(f"[Tracker] 相机启动失败 (可能被其他进程占用): {e}")
            self.is_running_flag = False
            return

        try:
            # 2. 检查全局的 stop_event，随时准备退出
            while not stop_event.is_set() and self.is_running_flag:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue

                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                color_image = np.asanyarray(color_frame.get_data())

                # --- YOLO & DeepSORT 推理 ---
                results = self.model(color_image, stream=True, verbose=False)
                detections = np.empty((0, 4))
                confarray = []
                
                for r in results:
                    for box in r.boxes:
                        if box.cls[0].int() == self.detect_class:
                            x1, y1, x2, y2 = box.xywh[0].int().tolist()
                            conf = round(box.conf[0].item(), 2)
                            detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                            confarray.append(conf)

                tracker_results = self.tracker.update(detections, confarray, color_image)
                
                frame_data = {"detections": []}
                nearest_det = None
                
                # --- 解析跟踪结果 ---
                for x1, y1, x2, y2, track_id in tracker_results:
                    ux, uy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    dis, cam_coord = self.get_3d_camera_coordinate([ux, uy], depth_frame, depth_intrin)
                    
                    if dis == 0: continue # 过滤无效深度
                    
                    det = {
                        "track_id": int(track_id),
                        "camera_coordinate": cam_coord # [x, y, z]
                    }
                    frame_data["detections"].append(det)

                    # 寻找目标逻辑：如果有指定目标找指定目标，否则找最近的
                    if self.target_track_id != -1:
                        if int(track_id) == self.target_track_id:
                            nearest_det = det
                    else:
                        if nearest_det is None or abs(cam_coord[2]) < abs(nearest_det["camera_coordinate"][2]):
                            nearest_det = det

                # --- 将结果通过 HTTP 发给 ROS 2 底盘 ---
                current_time = time.time()
                if nearest_det and (current_time - self.last_http_time > 0.1): # 10Hz发送频率
                    payload = {
                        "track_id": nearest_det["track_id"],
                        "x": nearest_det["camera_coordinate"][0],
                        "y": nearest_det["camera_coordinate"][1],
                        "z": nearest_det["camera_coordinate"][2],
                        "is_tracking": True
                    }
                    # 异步/非阻塞发送，防止 ROS2 挂了卡死视觉循环
                    threading.Thread(target=self._send_to_ros2, args=(payload,)).start()
                    self.last_http_time = current_time

                # --- 触发 LLM 播报回调 ---
                if llm_callback and frame_data["detections"]:
                    llm_callback(frame_data)

                # 本地调试显示
                if self.local_display:
                    if nearest_det:
                        cv2.putText(color_image, f"Target: ID {nearest_det['track_id']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Tracking", color_image)
                    cv2.waitKey(1)

        finally:
            # 3. 退出时必须释放相机，并通知 ROS2 停车
            print("[Tracker] 正在停止跟踪任务，释放相机...")
            pipeline.stop()
            if self.local_display:
                cv2.destroyAllWindows()
            self.is_running_flag = False
            self._send_to_ros2({"is_tracking": False})

    def _send_to_ros2(self, payload):
        try:
            requests.post(self.ros2_bridge_url, json=payload, timeout=0.2)
        except requests.exceptions.RequestException:
            pass # 忽略网络错误，避免刷屏