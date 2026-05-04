#!/usr/bin/env python3
"""
person_tracking_server.py
==========================
在 conda 环境中运行的人物跟踪服务端。
包装 PersonTrackingBridge，每帧将最近的人物坐标通过 HTTP POST 发给 ROS2 节点 (LLMlinkRos2V4)。

与 PersonTrackingBridge 的区别：
  - PersonTrackingBridge: 回调通知 LLMwithToolsV6 主线程（播报语音）
  - PersonTrackingServer: 额外通过 HTTP 将坐标发送给 LLMLinkRos2V4 节点，用于底盘跟随

用法 (在 LLMwithToolsV6 中):
    server = PersonTrackingServer()
    server.start(stop_event, llm_callback, target_track_id=-1)
    ...
    server.stop()
"""

import threading
import time
import requests
from typing import Optional, Callable, Dict, Any

from .person_tracking_llm_bridge import PersonTrackingBridge


class PersonTrackingServer:
    """
    人物跟踪服务端，在 conda 环境中运行。
    
    职责：
      1. 内部使用 PersonTrackingBridge 驱动 PersonCameraTracker
      2. 每帧提取最近的人，通过回调通知 LLM 主线程（播报语音）
      3. 每帧通过 HTTP POST 将坐标发送给 LLMLinkRos2V4 节点（底盘跟随）
    """

    def __init__(self,
                 ros2_bridge_url: str = "http://127.0.0.1:5000/person_track",
                 model_path: str = 'yolov8n.pt',
                 deepsort_checkpoint: str = 'person_tracking_ros/deep_sort/deep_sort/deep/checkpoint/ckpt.t7',
                 detect_class: int = 0,
                 save_point_cloud: bool = False,
                 point_cloud_path: str = 'point_cloud_data.txt',
                 local_display: bool = False):
        """
        参数:
            ros2_bridge_url: LLMLinkRos2V4 的 /person_track 路由地址
            model_path: YOLO 模型路径
            deepsort_checkpoint: DeepSort checkpoint 路径
            detect_class: 检测类别 (0=person)
            save_point_cloud: 是否保存点云
            point_cloud_path: 点云保存路径
            local_display: 是否弹出本地 cv2 窗口 (调试用)
        """
        self._ros2_bridge_url = ros2_bridge_url
        
        # 内部使用 PersonTrackingBridge 驱动相机
        self._bridge = PersonTrackingBridge(
            model_path=model_path,
            deepsort_checkpoint=deepsort_checkpoint,
            detect_class=detect_class,
            save_point_cloud=save_point_cloud,
            point_cloud_path=point_cloud_path,
            local_display=local_display,
        )
        
        self._stop_event: Optional[threading.Event] = None
        self._llm_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # 状态管理
        self._is_running = False
        self._last_send_time = 0.0
        self._send_interval = 0.1  # 10Hz 发送坐标给 ROS2

    # --- 公开接口 ---

    def start(self,
              stop_event: threading.Event,
              llm_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
              target_track_id: int = -1):
        """
        启动人物跟踪服务。
        
        参数:
            stop_event: 外部退出信号（全局 vision_stop_event）
            llm_callback: LLM 主线程回调（用于播报语音），可选
            target_track_id: 跟踪目标ID，-1=自动选择最近的人
        """
        if self._is_running:
            print("[TrackingServer] 已经在运行中，请先调用 stop()")
            return
        
        self._stop_event = stop_event
        self._llm_callback = llm_callback
        self._is_running = True
        
        # 定义内部回调：同时处理 LLM 播报和 ROS2 坐标发送
        def _on_frame(frame_data: Dict[str, Any]):
            self._handle_frame(frame_data)
        
        # 启动底层桥接器
        self._bridge.start(
            stop_event=stop_event,
            callback=_on_frame,
            target_track_id=target_track_id,
        )
        
        # 通知 ROS2 节点进入跟随模式
        self._notify_ros2_follow_start()
        
        print("[TrackingServer] 人物跟踪服务已启动，坐标将通过 HTTP 发送给 ROS2 节点")

    def stop(self):
        """停止跟踪服务并通知 ROS2 节点退出跟随模式。"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # 通知 ROS2 停止跟随
        self._notify_ros2_follow_stop()
        
        # 停止底层桥接器
        self._bridge.stop()
        
        print("[TrackingServer] 人物跟踪服务已停止")

    def is_running(self) -> bool:
        """返回跟踪服务是否正在运行。"""
        return self._is_running

    def set_target(self, track_id: int):
        """动态切换跟随目标。"""
        self._bridge.set_target(track_id)

    # --- 内部实现 ---

    def _handle_frame(self, frame_data: Dict[str, Any]):
        """
        每帧回调处理：
        1. 提取最近的人
        2. 调用 LLM 回调播报语音（如果提供）
        3. 通过 HTTP 发送坐标给 ROS2 节点
        """
        # 提取最近的人
        detections = frame_data.get("detections", [])
        nearest = None
        for det in detections:
            cam_coord = det.get("camera_coordinate", [0, 0, 0])
            cur_depth = abs(cam_coord[2])
            if nearest is None or cur_depth < abs(nearest["camera_coordinate"][2]):
                nearest = det
        
        if nearest is None:
            return
        
        # 1. LLM 回调（播报语音）
        if self._llm_callback:
            try:
                self._llm_callback(frame_data)
            except Exception as e:
                print(f"[TrackingServer] LLM 回调执行出错: {e}")
        
        # 2. 发送坐标给 ROS2（节流 10Hz）
        current_time = time.time()
        if current_time - self._last_send_time < self._send_interval:
            return
        self._last_send_time = current_time
        
        cam_coord = nearest["camera_coordinate"]
        track_id = nearest.get("track_id", -1)
        
        payload = {
            "track_id": track_id,
            "x": float(cam_coord[0]),
            "y": float(cam_coord[1]),
            "z": float(cam_coord[2]),
            "is_tracking": True,
        }
        
        try:
            resp = requests.post(self._ros2_bridge_url, json=payload, timeout=0.2)
            # 静默处理，避免刷屏
        except requests.exceptions.Timeout:
            pass  # 超时不打印，网络波动正常
        except requests.exceptions.ConnectionError:
            print("[TrackingServer] ⚠️ 无法连接到 ROS2 桥接节点 (LLMLinkRos2V4)，请确认节点已启动")
        except Exception as e:
            print(f"[TrackingServer] 发送跟踪数据失败: {e}")

    def _notify_ros2_follow_start(self):
        """通知 ROS2 节点进入人物跟随模式。"""
        try:
            requests.post(
                "http://127.0.0.1:5000/control",
                json={"name": "person_follow", "flag": "start"},
                timeout=2.0,
            )
            print("[TrackingServer] 已通知 ROS2 节点进入人物跟随模式")
        except Exception as e:
            print(f"[TrackingServer] ⚠️ 通知 ROS2 节点失败: {e}")

    def _notify_ros2_follow_stop(self):
        """通知 ROS2 节点退出人物跟随模式。"""
        try:
            requests.post(
                "http://127.0.0.1:5000/control",
                json={"name": "person_follow_stop", "flag": "start"},
                timeout=2.0,
            )
            print("[TrackingServer] 已通知 ROS2 节点退出人物跟随模式")
        except Exception as e:
            print(f"[TrackingServer] ⚠️ 通知 ROS2 节点停车失败: {e}")


# ============================
# 便捷工厂函数
# ============================

def create_tracking_server(**kwargs) -> PersonTrackingServer:
    """
    创建 PersonTrackingServer 实例的便捷工厂。
    接受与 PersonTrackingServer.__init__ 相同的参数。
    """
    return PersonTrackingServer(**kwargs)