#!/usr/bin/env python3
"""
person_tracking_llm_bridge.py
==============================
桥接模块：将纯视觉的 PersonCameraTracker 包装为 LLMwithToolsV6 可安全调用的线程工作器。

设计要点：
  - 通过 vision_stop_event 与其它视觉任务协调，避免 RealSense 相机争夺
  - 每帧检测结果通过 callback 异步反馈给主线程
  - 支持设置特写跟踪目标 (target_track_id)
  - 独立的模块，不依赖 LLMwithToolsV6 的内部实现细节

用法 (在 LLMwithToolsV6 中):
    tracker = PersonTrackingBridge(...)
    tracker.start(stop_event, callback, target_track_id=-1)
    ...
    tracker.stop()
"""

import threading
import time
from typing import Optional, Callable, Dict, Any

from .person_camera_tracker import PersonCameraTracker


class PersonTrackingBridge:
    """
    线程安全的人物跟踪桥接器。

    封装 PersonCameraTracker，提供 start/stop 接口以及每帧回调。
    通过外部传入的 threading.Event 实现与其他视觉任务的协同退出。
    """

    def __init__(self,
                 model_path: str = 'yolov8n.pt',
                 deepsort_checkpoint: str = 'person_tracking_ros/deep_sort/deep_sort/deep/checkpoint/ckpt.t7',
                 detect_class: int = 0,
                 save_point_cloud: bool = False,
                 point_cloud_path: str = 'point_cloud_data.txt',
                 local_display: bool = False):
        """
        参数:
            model_path: YOLO 模型路径
            deepsort_checkpoint: DeepSort checkpoint 路径
            detect_class: 检测类别 (0=person)
            save_point_cloud: 是否保存点云
            point_cloud_path: 点云保存路径
            local_display: 是否弹出本地 cv2 窗口 (调试用)
        """
        self._model_path = model_path
        self._deepsort_checkpoint = deepsort_checkpoint
        self._detect_class = detect_class
        self._save_point_cloud = save_point_cloud
        self._point_cloud_path = point_cloud_path
        self._local_display = local_display

        self._tracker: Optional[PersonCameraTracker] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

        # 运行时变量
        self._target_track_id: int = -1  # -1 = 跟随最近的
        self._callback: Optional[Callable[[Dict[str, Any]], None]] = None

    # --- 公开接口 ---

    def start(self,
              stop_event: threading.Event,
              callback: Callable[[Dict[str, Any]], None],
              target_track_id: int = -1):
        """
        启动人物跟踪后台线程。

        参数:
            stop_event: 外部传入的退出信号。当该 Event 被 set() 时，循环自动退出并释放相机。
            callback: 每帧结果回调函数。
                      签名: callback(frame_data: dict)
                      frame_data 是 PersonCameraTracker.process_frame() 返回的完整字典。
            target_track_id: 目标跟踪ID，-1 表示自动选择最近的人。
        """
        if self.is_running():
            print("[TrackingBridge] 已经在运行中，请先调用 stop()")
            return

        self._stop_event = stop_event
        self._callback = callback
        self._target_track_id = target_track_id

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[TrackingBridge] 人物跟踪已启动")

    def stop(self):
        """停止跟踪线程并释放相机资源。"""
        if self._thread and self._thread.is_alive():
            print("[TrackingBridge] 正在停止跟踪线程...")
            # 外部 stop_event 应该已经被 set，这里等待线程结束
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                print("[TrackingBridge] ⚠️ 跟踪线程未能在3秒内退出")
        self._thread = None
        self._tracker = None

    def is_running(self) -> bool:
        """返回跟踪是否正在运行。"""
        return self._thread is not None and self._thread.is_alive()

    def set_target(self, track_id: int):
        """动态切换跟随目标。"""
        self._target_track_id = track_id
        print(f"[TrackingBridge] 跟随目标已切换为 TrackID={track_id}")

    # --- 内部实现 ---

    def _run_loop(self):
        """后台线程主循环：初始化相机 → 逐帧处理 → 回调 → 检查退出信号。"""
        try:
            # 创建 PersonCameraTracker 实例（内部初始化 RealSense 相机）
            self._tracker = PersonCameraTracker(
                model_path=self._model_path,
                deepsort_checkpoint=self._deepsort_checkpoint,
                detect_class=self._detect_class,
                save_point_cloud=self._save_point_cloud,
                point_cloud_path=self._point_cloud_path,
            )

            while self._stop_event and not self._stop_event.is_set():
                frame_data = self._tracker.process_frame()
                if frame_data is None:
                    continue

                # 如果有指定 target_track_id，过滤结果
                if self._target_track_id >= 0:
                    filtered = [d for d in frame_data["detections"]
                                if d["track_id"] == self._target_track_id]
                    frame_data["detections"] = filtered

                # 通过回调将数据传给主线程
                if self._callback:
                    try:
                        self._callback(frame_data)
                    except Exception as e:
                        print(f"[TrackingBridge] 回调执行出错: {e}")

                # 可选：本地显示
                if self._local_display:
                    self._tracker.draw_detections(frame_data)
                    import cv2
                    cv2.imshow('LLM Person Tracking', frame_data["color_image"])
                    cv2.waitKey(1)

        except Exception as e:
            print(f"[TrackingBridge] 跟踪循环异常: {e}")
        finally:
            if self._tracker:
                self._tracker.cleanup()
                self._tracker = None
            print("[TrackingBridge] 跟踪线程已退出，相机资源已释放")


# ============================
# 便捷工厂函数
# ============================

def create_tracking_bridge(**kwargs) -> PersonTrackingBridge:
    """
    创建 PersonTrackingBridge 实例的便捷工厂。
    接受与 PersonTrackingBridge.__init__ 相同的参数。
    """
    return PersonTrackingBridge(**kwargs)