import time
import threading
import requests
import sys
import os

# === 导入你的视觉类 ===
# 请确保这里的路径与你实际的文件结构一致
from track.yolov8_deepsort_tracking.FaceAndBodyDetectV3 import AsyncVisionTracker

# === 模拟主控程序的全局变量和日志函数 ===
CodeInfo = "CodeInfo"
ROSInfo = "ROSInfo"

def print_info(role, msg):
    print(f"[{role}] {msg}")

# 全局停止信号
vision_stop_event = threading.Event()

# 模拟大模型的节流推送函数
def push_vision_to_llm(info):
    print_info("MOCK_LLM", f"🚨 [拦截到本该发给大模型的消息] -> {info}")

# 模拟停止当前视觉任务的函数
def stop_current_vision():
    print_info(CodeInfo, "🛑 触发自动停止机制，正在终止视觉线程...")
    vision_stop_event.set()

# === 核心：我们要测试的工作线程 ===
def run_person_track_worker():
    """
    独立测试版：人物对齐与跟踪线程
    """
    print_info(CodeInfo, "启动人物对齐与跟踪线程...")
    tracker = AsyncVisionTracker()  
    
    # 通知 ROS 2 底层准备接收追踪数据
    try:
        response = requests.post("http://127.0.0.1:5000/control", json={"name": "track", "flag": "start"}, timeout=2.0)
        print_info(ROSInfo, f"已通知底盘开启跟随模式: {response.json()}")
    except Exception as e:
        print_info(ROSInfo, f"⚠️ 通知 ROS 开始追踪失败，底盘节点是否未启动？: {e}")

    loop_count = 0
    
    try:
        # 当没有收到全局停止信号时，持续按 5Hz 循环
        while not vision_stop_event.is_set():
            start_time = time.time()
            data = tracker.get_control_data()
            
            if data["locked"]:
                face_off = data["face_offset"]
                body_off = data["body_offset"]
                
                abs_face = abs(face_off)
                abs_body = abs(body_off)
                
                is_aligned = False
                
                # =======================================================
                # 1. 终止条件判定：容错机制 (应对 -9999)
                # =======================================================
                # if abs_face < 6000 and abs_body < 6000:
                #     # 人脸和人体都在视野内：要求两者偏移都小于15
                #     is_aligned = (abs_face < 15 and abs_body < 15)
                # elif abs_face >= 6000 and abs_body < 6000:
                #     # 人脸丢失 (绝对值>6000)，只识别到人体：只要人体小于15即可
                #     is_aligned = (abs_body < 15)
                # elif abs_body >= 6000 and abs_face < 6000:
                #     # 人体丢失 (绝对值>6000)，只识别到人脸：只要人脸小于15即可
                #     is_aligned = (abs_face < 15)
                # else:
                #     # 都丢失了 (-9999)，显然没有对齐
                #     is_aligned = False

                # if is_aligned:
                #     print_info(CodeInfo, "✅ 顾客已完美对齐！停止旋转。")
                    
                #     # 发送停车指令
                #     try:
                #         requests.post("http://127.0.0.1:5000/control", json={"name": "person_follow_stop", "flag": "stop"}, timeout=2.0)
                #     except:
                #         pass
                        
                #     info = "我已经转身并面向了顾客，可以开始提供服务了。"
                #     push_vision_to_llm(info)
                    
                #     stop_current_vision()
                #     break

                # =======================================================
                # 2. 对齐阶段：选择有效的数据计算角速度 (Twist)
                # =======================================================
                # 优先用人体偏移进行控制，如果人体丢失，用人脸兜底
                active_offset = 0
                if abs_body < 6000:
                    active_offset = body_off
                elif abs_face < 6000:
                    active_offset = face_off
                else:
                    # 如果两个都变成了 -9999，active_offset 为 0
                    # 此时角速度为 0，底盘会原地悬停，等待顾客再次进入视野，非常安全
                    active_offset = 0 
                    
                Kp = 0.002  
                angular_z = -active_offset * Kp
                angular_z = max(-0.5, min(0.5, angular_z))
                
                twist_payload = {
                    "linear": {"x": 0.0, "y": 0.0, "z": 0.0}, 
                    "angular": {"x": 0.0, "y": 0.0, "z": angular_z}
                }
                
                try:
                    requests.post("http://127.0.0.1:5000/cmd_twist", json=twist_payload, timeout=0.1)
                except Exception:
                    pass
                    
                # =======================================================
                # 3. 控制打印频率为 1Hz
                # =======================================================
                if loop_count % 5 == 0:
                    face_str = "未识别" if abs_face >= 6000 else f"{face_off:>4}"
                    body_str = "未识别" if abs_body >= 6000 else f"{body_off:>4}"
                    print_info(CodeInfo, f"[底盘对齐中] 人脸偏移: {face_str} | 人体偏移: {body_str}")
                    # 将深度信息也打印出来，方便调试
                    print_info(CodeInfo, f"[底盘对齐中] 人脸深度: {data['face_depth']:.2f}m | 人体深度: {data['body_depth']:.2f}m")
                    
            else:
                if loop_count % 5 == 0:
                    print_info(CodeInfo, "[底盘对齐中] 视觉模块正在寻找目标，暂未锁定...")

            loop_count += 1
            
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.2 - elapsed))
            
    finally:
        tracker.release()
        print_info(CodeInfo, "对齐任务结束，相机资源已成功释放。")


# === 测试启动入口 ===
if __name__ == "__main__":
    print_info(CodeInfo, "================================================")
    print_info(CodeInfo, "🚀 独立视觉与底盘追踪测试程序已启动")
    print_info(CodeInfo, "提示: 请确保 LLMlinkRos2V6.py (端口5000) 已经在运行！")
    print_info(CodeInfo, "按 Ctrl+C 随时手动退出")
    print_info(CodeInfo, "================================================")
    
    # 启动工作线程
    track_thread = threading.Thread(target=run_person_track_worker, daemon=True)
    track_thread.start()
    
    try:
        # 主线程死循环维持程序运行，直到工作线程结束
        while track_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print_info(CodeInfo, "\n接收到键盘中断信号 (Ctrl+C)")
        stop_current_vision()
        track_thread.join(timeout=3.0)
    finally:
        print_info(CodeInfo, "测试程序完全退出。")