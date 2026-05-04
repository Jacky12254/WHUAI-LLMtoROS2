import os
import json
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel, Field
import threading
import queue
from flask import Flask, request, jsonify
# === ROS2 ===
import requests
#
import sys
import os
# 1. 获取当前代码运行的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 拼接出 deep_sort 的父级目录的绝对路径
# 注意：这里请核对一下你的文件夹名字是下划线 (_) 还是中划线 (-)
deepsort_path = os.path.join(current_dir, "person_tracking_ros", "deep_sort", "deep_sort", "deep", "checkpoint")
# 3. 将这个路径临时加入到 Python 的环境变量中
sys.path.append(deepsort_path)
#
# ===音频处理===
import subprocess 
import pygame
import time
from voice.voiceV6 import FunASRSpeechTranscriber  # 这是我们改进后的 FunASRRealtime 类所在的文件
import re
# === 视觉系统相关 ===
from hand_gesture.handV5 import start_gesture_recognition
from face_detect.face_detect_real_sense import recognize_people
from object_detect.detect_object import object_detection_loop
from track.yolov8_deepsort_tracking.FaceAndBodyDetectV3 import AsyncVisionTracker  # 这是我们改进后的视觉追踪类所在的文件

CodeInfo = "CodeInfo"
AIInfo = "AIInfo"
ROSInfo = "ROSIInfo"
def print_info(role:str,msg:str):
    print(f"{role}:{msg}")

def HTTP2ROS2(payload: dict, node: str) -> str:
    """将工具调用的指令通过 HTTP 发送给 ROS2 节点的函数。"""
    url = f"http://127.0.0.1:5000/{node}"
    response = requests.post(url, json=payload, timeout=2.0)
    return response.json()

# ================= 新增：全局视觉反馈节流阀 =================
global_vision_last_time = 0
tracking_frame_counter = 9  # 专门用于连续追踪的计数器
face_frame_counter = 9      # 专门用于人脸识别的计数器
is_first_vision_event = True  # 用于标记是否是第一次视觉事件，第一次事件不受节流限制

wave_person_3d = None  # 用于存储挥手时的人体空间坐标,作为全局变量让回调函数和主线程都能访问

def push_vision_to_llm(info, event_type="normal"):
    """
    统一管理视觉事件进入大模型队列的频率，防止大模型被垃圾信息淹没。
    event_type: 'normal' (按时间节流), 'tracking' (按频率/次数节流)
    """
    global global_vision_last_time, tracking_frame_counter
    current_time = time.time()
    
    # 1. 绝对静音原则：如果机器人正在说话，绝不塞入新视觉信息，避免自言自语打断
    if is_speaking.is_set():
        return

        # 2. 全局时间冷却：不管什么视觉事件，距离上一次通知大模型至少间隔 5 秒
        # （防止“看到人脸”、“识别物体”、“识别手势”在同一秒内疯狂挤入队列）
    if current_time - global_vision_last_time < 5.0:
        return

    if event_type == "recognize_people":
        global face_frame_counter
        face_frame_counter += 1
        if face_frame_counter < 10: 
            return
        face_frame_counter = 0

    # 4. 成功通过所有节流阀，放行！
    global_vision_last_time = current_time
    print_info(CodeInfo, f"[🚦 节流阀放行 -> 进入大脑]: {info}")
    # 记录第一次事件已经发生过了，以后都要受节流限制
    # 真正放入大模型的处理队列
    event_queue.put({"source": "vision_system", "content": info})
# ==========================================================

# --- 人脸识别回调 ---
last_face_time = {} # 记录每个人上次被打招呼的时间
def on_face_detected(name):
    current_time = time.time()
    # 如果是 unknown，或者5秒内刚打过招呼，就不理他
    if name == "unknown" or (current_time - last_face_time.get(name, 0) < 5.0):
        return
    if is_speaking.is_set(): return
    
    last_face_time[name] = current_time
    info = f"看到了名为 {name} 的人。"
    print_info(CodeInfo, f"[👀 视觉反馈]{info}")
    push_vision_to_llm(info, event_type="recognize_people")

def run_face_detect_worker():
    # 传入退出事件和回调函数
    recognize_people(stop_event=vision_stop_event, callback=on_face_detected)

# --- 物体检测回调 ---
last_obj_time = 0
def on_object_detected(obj_name, coordinates):
    global last_obj_time
    current_time = time.time()
    # 20秒冷却时间
    if current_time - last_obj_time < 20.0: return
    if is_speaking.is_set(): return
    
    last_obj_time = current_time
    info = f"在坐标 {coordinates} 处看到了物体：{obj_name}。"
    print_info(CodeInfo, f"[👀 视觉反馈]{info}")
    push_vision_to_llm(info)

def run_object_detect_worker():
    object_detection_loop(stop_event=vision_stop_event, callback=on_object_detected)

# --- 人物跟踪（基于 PersonTrackingServer，通过 HTTP 将坐标发给 ROS2 底盘） ---
# --- 人物跟踪（基于 AsyncVisionTracker，通过 HTTP 向 ROS2 发送 Twist 坐标） ---
def run_person_track_worker():
    """
    基于 AsyncVisionTracker 的后台跟踪线程。
    由大模型触发，5Hz 向 ROS 发送 Twist 控制旋转对齐，1Hz 打印终端，对齐后自动终止并向大模型反馈。
    """
    print_info(CodeInfo, "启动人物对齐与跟踪线程...")
    tracker = AsyncVisionTracker()  
    
    # 通知 ROS 2 底层准备接收追踪数据
    try:
        requests.post("http://127.0.0.1:5000/control", json={"name": "track", "flag": "start"}, timeout=2.0)
    except Exception as e:
        print_info(ROSInfo, f"⚠️ 通知 ROS 开始追踪失败: {e}")

    loop_count = 0
    
    try:
        # 当没有收到全局停止信号时，持续按 5Hz 循环
        #计时开始
        start_time_1=time.time()
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
                if abs_face < 6000 and abs_body < 6000:
                    # 人脸和人体都在视野内：要求两者偏移都小于15
                    is_aligned = (abs_face < 15 and abs_body < 15)
                elif abs_face >= 6000 and abs_body < 6000:
                    # 人脸丢失 (绝对值>6000)，只识别到人体：只要人体小于15即可
                    is_aligned = (abs_body < 15)
                elif abs_body >= 6000 and abs_face < 6000:
                    # 人体丢失 (绝对值>6000)，只识别到人脸：只要人脸小于15即可
                    is_aligned = (abs_face < 15)
                else:
                    # 都丢失了 (-9999)，显然没有对齐
                    is_aligned = False

                if is_aligned:
                    print_info(CodeInfo, "✅ 顾客已完美对齐，位于视野正中央！停止旋转。")
                    
                    # 给底盘发停止指令
                    try:
                        requests.post("http://127.0.0.1:5000/control", json={"name": "track", "flag": "stop"}, timeout=2.0)
                    except:
                        pass
                        
                    # 向大模型汇报情况，触发下一轮对话
                    info = "我已经转身并面向了顾客，可以开始提供服务了。"
                    push_vision_to_llm(info)
                    
                    # 从内部触发终止机制，安全杀掉自己这个视觉任务
                    stop_current_vision()
                    break

                # =======================================================
                # 2. 对齐阶段：选择有效的数据计算角速度 (Twist) 发给 ROS 2
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
                angular_z = max(-0.5, min(0.5, angular_z)) # 安全限幅 ±0.5 rad/s
                
                twist_payload = {
                    "linear": {"x": 0.0, "y": 0.0, "z": 0.0}, 
                    "angular": {"x": 0.0, "y": 0.0, "z": angular_z}
                }
                
                try:
                    # 发送 Twist 格式给底盘
                    requests.post("http://127.0.0.1:5000/cmd_twist", json=twist_payload, timeout=0.1)
                except Exception:
                    pass
                    
                # =======================================================
                # 3. 控制打印频率为 1Hz，防止消息挤满终端
                # =======================================================
                if loop_count % 5 == 0:
                    face_str = "未识别" if abs_face >= 6000 else f"{face_off:>4}"
                    body_str = "未识别" if abs_body >= 6000 else f"{body_off:>4}"
                    print_info(CodeInfo, f"[底盘对齐中] 人脸偏移: {face_str} | 人体偏移: {body_str}")
                
                current_time = time.time()
                if current_time - start_time_1 >10.0:
                    requests.post("http://127.0.0.1:5000/control", json={"name": "track", "flag": "stop"}, timeout=2.0)
                    info = "转身超时，但是可以开始提供服务了。"
                    push_vision_to_llm(info)
                    
                    # 从内部触发终止机制，安全杀掉自己这个视觉任务
                    stop_current_vision()
                    break
                    
            else:
                if loop_count % 5 == 0:
                    print_info(CodeInfo, "[底盘对齐中] 视觉模块正在寻找目标，暂未锁定...")

            loop_count += 1
            
            # 严格控制 5Hz 调用频率 (每次循环耗时补齐到 0.2 秒)
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.2 - elapsed))
            current_time = time.time()
            if current_time - start_time_1 >10.0:
                requests.post("http://127.0.0.1:5000/control", json={"name": "track", "flag": "stop"}, timeout=2.0)
                break
            
    finally:
        # 当 break 触发或者 vision_stop_event 被外部设置时，安全释放相机资源
        tracker.release()
        print_info(CodeInfo, "对齐任务结束，相机资源已成功释放。")

# ----------------- 补丁代码开始 -----------------
class PatchedChatOpenAI(ChatOpenAI):
    """
    自定义的 ChatOpenAI 类，用于修复本地模型违规返回字典格式 arguments 的兼容性问题。
    使用 *args 和 **kwargs 兼容不同版本的 LangChain 参数签名。
    """
    def _create_chat_result(self, response, *args, **kwargs):
        # 兼容对象和字典两种可能的数据格式
        choices = getattr(response, "choices", [])
        if not choices and isinstance(response, dict):
            choices = response.get("choices", [])

        for choice in choices:
            # 提取 message (兼容对象和字典)
            message = getattr(choice, "message", None) if not isinstance(choice, dict) else choice.get("message")
            
            if message:
                # 提取 tool_calls
                tool_calls = getattr(message, "tool_calls", []) if not isinstance(message, dict) else message.get("tool_calls", [])
                
                for tc in (tool_calls or []):
                    # 提取 function
                    func = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function")
                    
                    if func:
                        # 提取 arguments
                        args_val = getattr(func, "arguments", None) if not isinstance(func, dict) else func.get("arguments")
                        
                        # 【核心修复逻辑】：如果发现 arguments 是字典，将其序列化为 JSON 字符串
                        if isinstance(args_val, dict):
                            json_str = json.dumps(args_val, ensure_ascii=False)
                            if not isinstance(func, dict):
                                func.arguments = json_str
                            else:
                                func["arguments"] = json_str
                                
        # 将所有参数原封不动地传回给 LangChain 的原生逻辑
        return super()._create_chat_result(response, *args, **kwargs)
# ----------------- 补丁代码结束 -----------------

# ========= 定义工具输入输出格式和工具函数（使用 Pydantic 模型定义输入参数结构） ============
class GuestInfo(BaseModel):
    name: str = Field(description="客人次序，分为guest1和guest2")
    first_order: str = Field(description="客人下单的第一个菜品")
    second_order: str = Field(description="客人下单的第二个菜品")

@tool(args_schema=GuestInfo)
def RemenberTool(name: str, first_order: str, second_order: str) -> str:
    """用于记忆或查询客人信息的工具。请根据客人输入调用。"""
    # 模拟记忆逻辑
    return f"已记录：{name}点了{first_order}和{second_order}。【系统强制指令】：请立刻调用 Ros2ControlTool(name='goto_bar', flag='start') 前往吧台，不许做其他操作！"

class Ros2Node(BaseModel):
    '''定义发给ROS2节点的工具输入格式'''
    name: str = Field(description="机器人的程序名称,包括'goto_bar','goto_new_guest','goto_known_guest','grab','place'")
    flag: str = Field(description="控制标志，包括：'start' 或 'stop'")


@tool(args_schema=Ros2Node)
def Ros2ControlTool(name: str, flag: str) -> str:
    """用于控制 ROS2 机器人的工具。根据输入的程序名称和控制标志执行相应操作。"""
    # 模拟 ROS2 控制逻辑
    global wave_person_3d  # 声明使用全局变量，准备把挥手坐标发给 ROS2
    if name == 'goto_new_guest' and flag == 'start':
        stop_current_vision()  # 安全释放当前视觉任务
        payload = {
                "name": name,
                "flag": flag,
                "person_3d": list(wave_person_3d) if wave_person_3d else None  # 把挥手时的人体空间坐标也发给 ROS2，供它做更智能的决策
            }
    else:
        payload = {"name": name,
                "flag": flag,
                }
    if name == "grab" or name == "place":
        stop_current_vision()  # 安全释放当前视觉任务，确保机械臂操作时摄像头资源不冲突
    try:
        response = HTTP2ROS2(payload, node="control")
        print_info(ROSInfo, f"发送给 ROS 成功: {response}")
        status = "success"
        wave_person_3d = None  # 发送完指令后，重置挥手坐标，等待下一次挥手更新
        return(f"指令 [{name} - {flag}] 已成功下发至底层硬件。"
                f"【系统最高强制指令】：机器人的物理动作正在执行中！"
                f"你现在**必须停止**调用任何新的动作工具（绝对禁止紧接着调用 grab、place 或 goto）。"
                f"请仅用语言告知客人你正在前往或正在操作，然后立刻结束对话思考。"
                f"你必须静静等待底层系统发来 'Successfully reached' 或 'completed successfully' 的硬件反馈后，才能决定下一步行动！")
    except Exception as e:
        print_info(ROSInfo, f"发送失败，ROS 节点启动了吗？{e}")
        status = "failed"

# === 新增：视觉系统状态管理 ===
vision_stop_event = threading.Event()  # 用于通知当前视觉线程退出的信号
current_vision_task = "none"           # 记录当前的视觉任务
vision_thread = None                   # 新增：记录当前的视觉线程对象，用于追踪其生死

def stop_current_vision():
    """安全停止当前视觉任务，死等硬件被彻底释放"""
    global vision_thread, current_vision_task

    # 1. 【加固】：先无脑发出停止信号
    vision_stop_event.set()
        
    # 2. 如果旧线程还在跑，要求它退出并等待
    if vision_thread is not None and vision_thread.is_alive():
        # 👇 【核心修复】：防止“自己等自己死”的 RuntimeError
        if threading.current_thread() == vision_thread:
            print_info(CodeInfo, f"🛑 视觉任务 [{current_vision_task}] 正在由内部自行终止释放...")
        else:
            print_info(CodeInfo, f"🛑 正在从外部停止任务 [{current_vision_task}]，等待摄像头底层安全释放...")
            
            # 只有当外部线程（比如主线程）来调这个函数时，才死等它退出
            vision_thread.join(timeout=5.0) 
            
            if vision_thread.is_alive():
                print_info(CodeInfo, "⚠️ 警告：旧视觉线程未能及时退出，可能会引发资源冲突！")
            else:
                print_info(CodeInfo, "✅ 摄像头硬件已成功释放！")
            
    # 3. 硬件释放完毕后，安全重置开关
    vision_stop_event.clear() 
    current_vision_task = "none"

class VisionCommand(BaseModel):
    task: str = Field(description="视觉任务名称，可选值：'hand', 'stop'")

@tool(args_schema=VisionCommand)
def VisionControlTool(task: str) -> str:
    """当客人要求你认人、看物体、跟随/跟踪某人、或者识别手势时，必须调用此工具开启相应的视觉摄像头任务。"""
    global current_vision_task, vision_thread
    
    if task == current_vision_task:
        return f"视觉任务 {task} 已经在运行中了。"

    # 1. 安全杀掉旧视觉线程，并释放摄像头
    stop_current_vision()
    
    # 2. 根据大模型的指令，开启新的子线程，并记录到 vision_thread
    current_vision_task = task
    if task == 'face':
        vision_thread = threading.Thread(target=run_face_detect_worker, daemon=True)
        vision_thread.start()
        return "已开启人脸识别模式，正在扫描..."
    elif task == 'hand':
        vision_thread = threading.Thread(target=vision_listener_worker, daemon=True)
        vision_thread.start()
        return "已开启手势识别模式..."
    elif task == 'guest':
        vision_thread = threading.Thread(target=run_person_track_worker, daemon=True)
        vision_thread.start()
        return "已开启人物跟踪模式，正在寻找目标..."
    elif task == 'stop':
        return "已关闭所有摄像头视觉任务。"
    else:
        return f"未知的视觉任务: {task}"

tools = [RemenberTool, Ros2ControlTool, VisionControlTool]

# =============工具定义结束，下面是主程序和全局变量=============

is_speaking = threading.Event()
is_processing = threading.Event()
last_speak_end_time = 0.0  # 记录上一次说话结束的具体时间
class LLMwithTools:
    def __init__(self):
        """启动llm和工具"""
        self.output = "response.wav"
        self.piper_model_path = "/home/jacky/vision/voice/models/en_US-lessac-medium.onnx"
        self.piper_exe_path = "/home/jacky/vision/piper/piper"
        pygame.mixer.init(frequency=22050, size=-16, channels=1)

        # 注意：这里替换成我们打过补丁的 PatchedChatOpenAI
        self.client = PatchedChatOpenAI(
            base_url="http://127.0.0.1:8080",
            api_key="EMPTY",
            model="Qwen3.5-9B",
            temperature=0.7,
        )
        self.llm_with_tools = self.client.bind_tools(tools, tool_choice="auto")
        self.conversation_history = []  # 用于存储对话历史
        self.response = None
    
    def chat(self, messages):
        """聊天接口，输入消息列表，返回回复"""
        self.response = self.llm_with_tools.invoke(messages)
        return self.response
    
    def history(self):
        """历史记录接口，返回历史消息列表"""
        return self.conversation_history
    
    # ---------- 语音合成和播放相关函数 ----------
    def speakout(self, response): 
        if not response or not response.strip(): 
            return False
            
        # 清理文本，防止特殊字符干扰
        clean_text = re.sub(r'[^\w\s.,!?\'"-]', '', response).strip()
        
        if not clean_text:
            print_info(CodeInfo, "⚠️ 警告：大模型输出的内容被净化后为空！跳过合成。")
            return False
            
        print_info(CodeInfo, f"准备使用 Piper CLI 合成: {clean_text}")
        
        try:
            # 每次合成前先删除旧的残次品，防止 Pygame 读错
            if os.path.exists(self.output):
                os.remove(self.output)

            process = subprocess.Popen(
                [self.piper_exe_path, '--model', self.piper_model_path, '--output_file', self.output],
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            # 将清洗后的文本编码并推入命令行工具
            stdout_data, stderr_data = process.communicate(input=clean_text.encode('utf-8'))
            
            if process.returncode != 0:
                print_info(CodeInfo, f"❌ Piper CLI 执行失败: {stderr_data.decode('utf-8', errors='ignore')}")
                return False
                
            # 检查文件是否真的生成，且大小大于 44 字节（排除空壳）
            if os.path.exists(self.output) and os.path.getsize(self.output) > 44:
                return True
            else:
                print_info(CodeInfo, "❌ Piper 静默失败：生成的 WAV 文件没有音频数据！")
                return False
                
        except Exception as e:
            print_info(CodeInfo, f"❌ 语音合成发生代码异常: {e}")
            return False
        
    def play_audio(self):
        '''播放合成的音频，并在播放期间设置说话状态，结束后更新最后说话时间。'''
        global last_speak_end_time  # <--- 新增：声明使用全局变量  
        
        pygame.mixer.music.load(self.output)
        time.sleep(0.5)
        
        # 说话前，把锁拉上（捂住耳朵）
        is_speaking.set() 
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            continue
            
        time.sleep(0.5)
        # 说话结束，把锁打开（松开耳朵）
        is_speaking.clear()
        
        # <--- 新增：记录锁刚刚打开的精确时间
        last_speak_end_time = time.time()

    def chatandspeakout(self, response):   
        if response.content:
            print_info(AIInfo, response.content)
            self.speakout(response.content)
            self.play_audio()
        return response

    # 注意：这里增加了一个 execution_history 参数，用于记录当前轮次执行过的工具
    def check_tool_calls(self, response, messages, execution_history=None):
        """处理工具调用，支持连续多次递归调用，并自带死循环拦截功能"""
        if execution_history is None:
            execution_history = set()  # 初始化记录本

        if response.tool_calls:
            print_info(CodeInfo, f"触发工具调用: {response.tool_calls}")
            
            if hasattr(response, 'content') and response.content:
                print_info(AIInfo, f"[工具执行前的话术]: {response.content}")
                self.speakout(response.content)
                self.play_audio()
                
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                tool_result = "未知工具执行失败"

                # 🛑 【终极防线】：死循环拦截器
                if tool_name in execution_history:
                    tool_result = f"【系统严重警告】：拦截到死循环！你刚才已经调用过 {tool_name} 了，严禁重复调用同一个工具！请立刻闭嘴结束思考，或者调用 SOP 中的下一个动作工具（如 goto_bar）。"
                    print_info(CodeInfo, f"🚫 成功拦截大模型死循环重复调用: {tool_name}")
                else:
                    # 正常执行工具，并记录到历史中
                    execution_history.add(tool_name)
                    
                    if tool_name == "RemenberTool":
                        tool_result = RemenberTool.invoke(tool_args)
                        print_info(CodeInfo, f"🔧 记忆工具执行结果: {tool_result}")
                    elif tool_name == "Ros2ControlTool":
                        tool_result = Ros2ControlTool.invoke(tool_args)
                        print_info(CodeInfo, f"🤖 ROS2 工具执行结果: {tool_result}")
                    elif tool_name == "VisionControlTool":
                        tool_result = VisionControlTool.invoke(tool_args)
                        print_info(CodeInfo, f"👁️ 视觉控制工具执行结果: {tool_result}")

                messages.append(ToolMessage(
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                    content=str(tool_result)
                ))
            
            print_info(CodeInfo, "工具执行完毕，正在请求大模型进行下一步决策...")
            next_response = self.chat(messages)
            messages.append(next_response)
            
            # 递归调用时，把“记录本”传下去，防止它在下一层继续犯病
            self.check_tool_calls(next_response, messages, execution_history)
            
        else:
            self.chatandspeakout(response)

# === 2. 新增：无意义字符串过滤函数 ===
def is_meaningful(text):
    # 1. 基础清理
    text = text.strip()
    
    # 2. 剔除所有标点符号，只保留字母、数字和空格，然后转小写
    # 比如 ", thats.     " 会变成 "thats", "I." 会变成 "i"
    clean_text = re.sub(r'[^\w\s]', '', text).strip().lower()
    
    # 3. 长度过滤：基于清理后的纯字母判断
    # 这样 "I." 变成 "i" 后，长度为 1，就会被成功拦截
    if len(clean_text) <= 1:
        return False
        
    # 4. 无意义词库 (注意：这里必须全部使用小写，且不要加标点符号)
    useless_words = [
        "yeah", "what", "okay", "the", "um", "uh", "hello", 
        "hi", "test", "i", "oh", "thats", "thank"
    ]
    
    if clean_text in useless_words:
        return False
        
    return True


if __name__ == "__main__":
    llm = LLMwithTools()
    
    # === 1. 创建全局事件队列 ===
    event_queue = queue.Queue()
    voice_transcriber = FunASRSpeechTranscriber()
    is_processing.set()  # 启动时先锁住，等系统完全就绪后再解锁
    # === 改造 1：语音监听线程 ===
    def voice_listener_worker():
        global last_speak_end_time 
        
        while True:
            if is_processing.is_set():
                time.sleep(0.1) 
                continue
                
            # 【将锁传进去】：让麦克风随时能被其他事件打断
            user_input = voice_transcriber.get_next_utterance(interrupt_event=is_processing)

            
            # --- 加入这行调试代码 ---
            # print_info("CodeInfo", f"[DEBUG 麦克风捕获原始内容]: '{user_input}'")
            # -------------------------
            
            # 过滤掉被中途打断返回的空字符，以及残音
            if not user_input or not user_input.strip(): continue
            if time.time() - last_speak_end_time < 1.5: continue # 冷却时间拉长到1.5秒，防止录到回音
            if not is_meaningful(user_input): continue
                
            # 在交给主线程之前，子线程自己先把麦克风系统锁住！
            is_processing.set() 
            event_queue.put({"source": "human", "content": user_input})

    threading.Thread(target=voice_listener_worker, daemon=True).start()

    # === 3. 启动 ROS2 反馈监听服务器 (Flask) 线程 ===
    # 假设你的大模型跑在 5001 端口监听 ROS2 的反馈
    ros_app = Flask(__name__)
    
    @ros_app.route('/ros2_feedback', methods=['POST'])
    def receive_ros2_feedback():
        data = request.json
        info = data.get('info', '')
        if info:
            print_info(ROSInfo, f"[🔔 收到 ROS2 底层反馈]: {info}")
            
            # 核心修改：到达目标后，系统自动开启人脸人体对齐
            if "Successfully reached the target point" in info:
                print_info(CodeInfo, "📍 已到达目标点，系统自动开启人脸人体对齐工具...")
                VisionControlTool.invoke({"task": "guest"})
                info += " The system has automatically started face and body alignment tracking."

            # 将硬件反馈事件放入队列
            event_queue.put({"source": "robot_system", "content": info})
        return jsonify({"status": "received"}), 200

    def ros_server_worker():
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR) # 关闭 flask 烦人的默认输出
        ros_app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

# ... [保留原有的 ros_server_worker 启动代码] ...
    threading.Thread(target=ros_server_worker, daemon=True).start()

    # === 3.5 新增：启动视觉/手势监听线程 ===
    # 添加一个全局变量用于控制大模型处理挥手动作的冷却时间（防止摄像头疯狂抓取导致大模型被撑爆）
    last_vision_trigger_time = 0 

    def on_wave_detected(wave_dir = None, distance=None, person_3d=None):
        global last_vision_trigger_time, wave_person_3d
        current_time = time.time()
        
        # 5秒防抖冷却
        if current_time - last_vision_trigger_time < 5.0: return
        if is_speaking.is_set(): return
            
        last_vision_trigger_time = current_time
        
        # 组装带空间感知的环境提示词
        location_desc = ""
        if person_3d is not None:
            px, py, pz = person_3d
            wave_person_3d = person_3d  # 更新全局变量，供主线程使用
            
            # 在 RealSense 坐标系中，X的正数代表右边，负数代表左边 (单位是米)
            # 我们可以设置一个 0.15 米 (15厘米) 的阈值来判断客人是不是在正前方
            if px < -0.15:
                dir_str = "你的左前方"
            elif px > 0.15:
                dir_str = "你的右前方"
            else:
                dir_str = "你的正前方"
                
            location_desc = f"在距离你 {pz:.2f} 米的{dir_str} (精确空间坐标为: X={px:.2f}, Y={py:.2f}, Z={pz:.2f})，"
            
        elif distance and distance > 0.0:
            # 如果深度没抓好，保底只播报距离
            location_desc = f"在距离你 {distance:.2f} 米的地方，"
        
        # 将空间坐标发送给ros2
        payload = {"name": "wave_detect",
                "flag": "start",
                "person_3d": list(wave_person_3d) if wave_person_3d else None}
        try:
            response = HTTP2ROS2(payload, node="control")
            print_info(ROSInfo, f"挥手发送给 ROS 成功: {response}")
        except Exception as e:
            print_info(ROSInfo, f"挥手发送失败，ROS 节点启动了吗？{e}")

        info = f"{location_desc}客人正在向你挥手打招呼。"
        print_info(CodeInfo, f"[👀 收到视觉传感器反馈]: {info}")
        
        # 将带有精确坐标的视觉事件放入全局队列，LLM会在主循环中读取
        push_vision_to_llm(info)

    # 修改 on_point_detected，增加 target_3d 参数
    def on_point_detected(tip_3d, vector_3d, target_3d):
        global last_vision_trigger_time
        current_time = time.time()
        
        if current_time - last_vision_trigger_time < 5.0: return
        if is_speaking.is_set(): return
            
        last_vision_trigger_time = current_time
        
        tx, ty, tz = target_3d
        info = f"客人正用手指指向一个物体！该物体的精确三维坐标为: ({tx:.2f}, {ty:.2f}, {tz:.2f}) (单位:米)。请根据这个坐标控制机械臂或底盘去获取它。"
            
        print_info(CodeInfo, f"[🎯 收到指向反馈]: {info}")
        push_vision_to_llm(info)

    def vision_listener_worker():
        # 【修改】：将两个回调函数以及 stop_event 都传给底层
        start_gesture_recognition(
            on_wave_callback=on_wave_detected, 
            # on_point_callback=on_point_detected, 
            stop_event=vision_stop_event,
            local_display=True
        )

    # 启动视觉识别守护线程
    current_vision_task = "hand"
    vision_thread=threading.Thread(target=vision_listener_worker, daemon=True)
    vision_thread.start()
    # ==================================


    # === 4. 初始化对话 ===
    system_prompt = """You are an Embodied AI Robot Waiter competing in a professional restaurant service task. You must speak entirely in English.

【YOUR CORE MISSION & WORKFLOW】
You must complete exactly TWO customer orders in total. Follow this strict chronological workflow:

--- Order 1 ---
1. The system automatically initialized wave detection. Wait for the customer to wave.
2. When the customer waves, immediately use Ros2ControlTool(name="goto_new_guest", flag="start") to navigate to them. The system will process the coordinates, record the path, and automatically start face/body alignment when you arrive.
3. Once alignment is complete, greet the customer and ask what they want to order (they need 1 food, 1 drink).
4. CRITICAL CONFIRMATION: You MUST verbally repeat the items back to the customer and ask for confirmation (e.g., "You want a cola and a bread, is that correct?"). DO NOT USE the RemenberTool YET.
5. Once the customer explicitly says "Yes" or confirms, use RemenberTool(name="guest1", ...) to save the order.
6. After saving memory successfully, use Ros2ControlTool(name="goto_bar", flag="start") to go to the bar.
7. Speak to the bartender to place the order clearly and ask the bartender for the items.
8. Deliver the items by returning to the customer using Ros2ControlTool(name="goto_known_guest", flag="start") and ask the customer to take them.
9. When the FIRST order completes its LAST "place" action, you MUST autonomously use Ros2ControlTool(name="goto_bar", flag="start") to return to the bar.

--- Order 2 ---
10. Upon arriving at the bar after finishing Order 1, you MUST trigger VisionControlTool(task="hand") to start wave detection again.
11. Wait for the second customer to wave, then repeat the exact same process (Navigate -> Ask -> Confirm -> RemenberTool(name="guest2", ...) -> Go to Bar -> Deliver).
12. After the LAST "place" action for the SECOND order, autonomously use Ros2ControlTool(name="goto_bar", flag="start") to return to the bar. 
13. Arriving at the bar this time means all tasks are finished.

【RULES & CONSTRAINTS】
- Food includes: biscuit, chip, lays, bread, cookies; Drink includes: water, sprite, cola, orange, milk. Do fuzzy matching but confirm before moving!
- NEVER ask humans for navigation help.
- PARALLEL ACTIONS: Tell the user what you are doing while using a tool. (e.g., "I've noted your order and will head to the bar right now.").
- NEVER use RemenberTool before explicit verbal confirmation.
    
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="现在简单做一个开场白，只需要说一句话，系统已自动开启wave识别，等待顾客出现并打招呼。")
    ]
    response = llm.chat(messages)
    response = llm.chatandspeakout(response)
    messages.append(response)

# === 5. 主循环：处理事件队列 ===
    # === 改造 2：主循环 ===
    print_info(CodeInfo, "\n>>> 系统已就绪！可以说话，或等待 ROS2 发送事件...")
    is_processing.clear()  # 系统完全就绪，解锁麦克风环境监听
    try:
        while True:
            event = event_queue.get() 
            
            # 确保只要进入处理状态，锁必须是拉上的（应对视觉和ROS2挤进来的事件）
            is_processing.set() 
            # 给底层 100ms 的喘息时间，确保麦克风流在 C 语言层面彻底断开，绝对不跟 pygame 抢
            time.sleep(0.1) 
            
            try:
                if event["source"] == "human":
                    print_info(CodeInfo, f"客人说: {event['content']}")
                    messages.append(HumanMessage(content=event["content"]))

                elif event["source"] == "robot_system":
                    print_info(CodeInfo, f"[🔔 收到系统反馈]: {event['content']}")
                    prompt = f"【系统提示:来自底层传感器的反馈】： {event['content']},根据当前的任务和接下来的任务合理安排你的动作，并简短地告诉客人你接下来的动作计划。"
                    messages.append(HumanMessage(content=prompt))
                
                elif event["source"] == "vision_system":
                    prompt = f"【系统提示：来自视觉传感器的反馈】: {event['content']}。请严格根据你的 Workflow 继续下一步服务！"
                    messages.append(HumanMessage(content=prompt))

                # 交给大模型处理并播报
                response = llm.chat(messages)
                messages.append(response)
                llm.check_tool_calls(response, messages)
                
            finally:
                # 所有事情（包括发声、调用工具）全部干完后，才统一解锁
                print_info(CodeInfo, "\n[系统空闲] 重新开启麦克风环境监听...")
                is_processing.clear()
                
    except KeyboardInterrupt:
        print_info(CodeInfo, "系统正在退出...")
    finally:
        voice_transcriber.close()