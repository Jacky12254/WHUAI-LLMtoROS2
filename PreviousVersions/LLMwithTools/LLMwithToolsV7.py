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
import pyttsx3
from scipy import signal   # 用于重采样
import edge_tts
import asyncio
import pygame
import time
from voice.voiceV5 import FunASRSpeechTranscriber  # 这是我们改进后的 FunASRRealtime 类所在的文件
import re
# === 视觉系统相关 ===
from hand_gesture.handV4 import start_gesture_recognition
from face_detect.face_detect_real_sense import recognize_people
from object_detect.detect_object import object_detection_loop
from person_tracking_ros.person_tracking_serverV2 import PersonTrackingServer

CodeInfo = "CodeInfo"
AIInfo = "AIInfo"
ROSInfo = "ROSIInfo"
def print_info(role:str,msg:str):
    print(f"{role}:{msg}")
    
# ================= 新增：全局视觉反馈节流阀 =================
global_vision_last_time = 0
tracking_frame_counter = 9  # 专门用于连续追踪的计数器
face_frame_counter = 9      # 专门用于人脸识别的计数器
is_first_vision_event = True  # 用于标记是否是第一次视觉事件，第一次事件不受节流限制

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
            
        # 3. 针对人物跟踪的“按次数/帧数”特殊节流（每收到 10 次有效的追踪帧，才考虑放行 1 次）
    if event_type == "tracking":
        tracking_frame_counter += 1
        if tracking_frame_counter < 10: 
            return
            # 攒够 10 次了，重置计数器
        tracking_frame_counter = 0 

    if event_type == "recognize_people":
        global face_frame_counter
        face_frame_counter += 1
        if face_frame_counter < 10: 
            return
        face_frame_counter = 0

    # 4. 成功通过所有节流阀，放行！
    global_vision_last_time = current_time
    print_info(CodeInfo, f"[🚦 节流阀放行 -> 进入大脑]: {info}")
    # is_first_vision_event = False  # 记录第一次事件已经发生过了，以后都要受节流限制
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
    # event_queue.put({"source": "vision_system", "content": info})
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
    # event_queue.put({"source": "vision_system", "content": info})
    push_vision_to_llm(info)

def run_object_detect_worker():
    object_detection_loop(stop_event=vision_stop_event, callback=on_object_detected)

# --- 人物跟踪（基于 PersonTrackingServer，通过 HTTP 将坐标发给 ROS2 底盘） ---
tracking_server = PersonTrackingServer(
    ros2_bridge_url="http://127.0.0.1:5000/person_track",
    model_path='yolov8n.pt',
    deepsort_checkpoint='person_tracking_ros/deep_sort/deep_sort/deep/checkpoint/ckpt.t7',
    detect_class=0,
    save_point_cloud=False,
    local_display=False,
)
last_track_time = 0

def on_person_tracking_frame(frame_data):
    """
    人物跟踪每帧回调：接收 PersonCameraTracker.process_frame() 的完整结果。
    从检测结果中自动选出最近的人，按节流频率向主线程报告状态。
    """
    global last_track_time
    current_time = time.time()
    # 10秒播报一次跟踪状态
    if current_time - last_track_time < 10.0:
        return
    if is_speaking.is_set():
        return

    # 自动选择最近的人 (z 最小)
    detections = frame_data.get("detections", [])
    nearest = None
    for det in detections:
        cam_coord = det["camera_coordinate"]
        cur_depth = abs(cam_coord[2])
        if nearest is None or cur_depth < abs(nearest["camera_coordinate"][2]):
            nearest = det

    if nearest:
        last_track_time = current_time
        cam_coord = nearest["camera_coordinate"]
        depth = abs(cam_coord[2])
        track_id = nearest["track_id"]
        info = (f"正在跟踪目标人物(ID:{track_id})，"
                f"距离 {depth:.2f} 米，"
                f"坐标 ({cam_coord[0]:.2f}, {cam_coord[1]:.2f}, {cam_coord[2]:.2f})。")
        print_info(CodeInfo, f"[👀 视觉反馈]{info}")
        # event_queue.put({"source": "vision_system", "content": info})
        push_vision_to_llm(info, event_type="tracking")

def run_person_track_worker():
    """基于 PersonTrackingServer 的后台跟踪线程（每帧通过 HTTP 发送坐标给 ROS2 节点）。"""
    tracking_server.start(
        stop_event=vision_stop_event,
        llm_callback=on_person_tracking_frame,
        target_track_id=-1,  # 自动跟踪最近的
    )

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

class GuestInfo(BaseModel):
    name: str = Field(description="客人姓名")
    preference: str = Field(description="客人喜好")
    current_place: list[str] = Field(default_factory=list, description="当前位置列表")

@tool(args_schema=GuestInfo)
def RemenberTool(name: str, preference: str, current_place: list[str]) -> str:
    """用于记忆或查询客人信息的工具。请根据客人输入调用。"""
    # 模拟记忆逻辑
    return f"已记录：{name}喜欢{preference}，在{current_place}"

class Ros2Node(BaseModel):
    '''定义发给ROS2节点的工具输入格式'''
    name: str = Field(description="机器人的程序名称,包括“move”、“takebag”")
    flag: str = Field(description="控制标志，例如：'start' 或 'stop'")

@tool(args_schema=Ros2Node)
def Ros2ControlTool(name: str, flag: str) -> str:
    """用于控制 ROS2 机器人的工具。根据输入的程序名称和控制标志执行相应操作。"""
    # 模拟 ROS2 控制逻辑
    url = "http://127.0.0.1:5000/control"
    payload = {"name": name,
                "flag": flag}
    try:
        response = requests.post(url, json=payload, timeout=2.0)
        print_info(ROSInfo, f"发送给 ROS 成功: {response.json()}")
        status = "success"
    except Exception as e:
        print_info(ROSInfo, f"发送失败，ROS 节点启动了吗？{e}")
        status = "failed"

    return f"ROS2 控制指令：{name} - {flag} - {status}"

# === 新增：视觉系统状态管理 ===
vision_stop_event = threading.Event()  # 用于通知当前视觉线程退出的信号
current_vision_task = "none"           # 记录当前的视觉任务
vision_thread = None                   # 新增：记录当前的视觉线程对象，用于追踪其生死

def stop_current_vision():
    """安全停止当前视觉任务，死等硬件被彻底释放"""
    global vision_thread, current_vision_task
    
    # 1. 业务层先发停止指令
    if current_vision_task == "track":
        tracking_server.stop()

    # 【加固】：不管三七二十一，先无脑发出停止信号
    vision_stop_event.set()
        
    # 2. 如果旧线程还在跑，强制要求它退出并等待它死透
    if vision_thread is not None and vision_thread.is_alive():
        print_info(CodeInfo, f"🛑 正在停止任务 [{current_vision_task}]，等待摄像头底层安全释放...")
        vision_stop_event.set()
        
        # 关键修正：死等旧线程执行完 finally: pipeline.stop()，最多等5秒
        vision_thread.join(timeout=5.0) 
        
        if vision_thread.is_alive():
            print_info(CodeInfo, "⚠️ 警告：旧视觉线程未能及时退出，可能会引发资源冲突！")
        else:
            print_info(CodeInfo, "✅ 摄像头硬件已成功释放！")
            
    # 3. 硬件释放完毕后，安全重置开关
    vision_stop_event.clear() 
    current_vision_task = "none"

class VisionCommand(BaseModel):
    task: str = Field(description="视觉任务名称，可选值：'face', 'object', 'track', 'hand', 'stop'")

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
    elif task == 'object':
        vision_thread = threading.Thread(target=run_object_detect_worker, daemon=True)
        vision_thread.start()
        return "已开启物体检测模式，正在寻找物体..."
    elif task == 'track':
        vision_thread = threading.Thread(target=run_person_track_worker, daemon=True)
        vision_thread.start()
        return "已开启人物跟踪模式（底盘跟随已激活）..."
    elif task == 'hand':
        vision_thread = threading.Thread(target=vision_listener_worker, daemon=True)
        vision_thread.start()
        return "已开启手势识别模式..."
    elif task == 'stop':
        return "已关闭所有摄像头视觉任务。"
    else:
        return f"未知的视觉任务: {task}"


class TrackCommand(BaseModel):
    action: str = Field(description="操作：'start' (开始跟踪), 'stop' (停止跟踪), 'switch_target' (切换ID)")
    track_id: int = Field(default=-1, description="目标track_id")

@tool(args_schema=TrackCommand)
def PersonTrackTool(action: str, track_id: int = -1) -> str:
    """专门用于人物跟踪的工具。"""
    global current_vision_task, vision_thread

    if action == 'start':
        if tracking_server.is_running():
            return "人物跟踪已经在运行中。"
            
        stop_current_vision() # 安全释放
        
        current_vision_task = "track"
        vision_thread = threading.Thread(target=run_person_track_worker, daemon=True)
        vision_thread.start()
        return "已开启人物跟踪模式，正在跟随最近的人，底盘已激活。"

    elif action == 'stop':
        if not tracking_server.is_running():
            return "人物跟踪未在运行。"
            
        stop_current_vision() # 安全释放跟踪线程
        
        # 自动恢复挥手识别
        print_info(CodeInfo, "✅ 人物跟踪已安全结束，正在恢复挥手识别...")
        current_vision_task = "hand"
        vision_thread = threading.Thread(target=vision_listener_worker, daemon=True)
        vision_thread.start()
        
        return "已停止人物跟踪，底盘已停车，挥手识别已恢复。"

    elif action == 'switch_target':
        if not tracking_server.is_running():
            return "人物跟踪未在运行，请先启动跟踪。"
        tracking_server.set_target(track_id)
        return f"已切换到跟踪目标 ID={track_id}。"
    else:
        return f"未知的人物跟踪操作：{action}"
tools = [RemenberTool, Ros2ControlTool, VisionControlTool, PersonTrackTool]
is_speaking = threading.Event()
is_processing = threading.Event()

last_speak_end_time = 0.0  # 记录上一次说话结束的具体时间
class LLMwithTools:
    def __init__(self):
        """启动llm和工具"""

        self.engine = pyttsx3.init()
        self.rate = '-4%'
        self.volume = '+0%'
        self.output = "response.mp3"
        self.sound_voice = 'en-US-AriaNeural'

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
    
    # === 发出声音 ===
    async def speakout(self, response): 
        # 这里可以添加文本转语音的代码，例如调用 TTS 模型或系统 TTS 功能
        # 例如，使用 pyttsx3 库：
        tts = edge_tts.Communicate(text=response, voice=self.sound_voice, rate=self.rate, volume=self.volume)
        await tts.save(self.output)

    def play_audio(self):
        global last_speak_end_time  # <--- 新增：声明使用全局变量
        
        pygame.mixer.init()
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
            asyncio.run(self.speakout(response.content))
            self.play_audio()
        return response

    def check_tool_calls(self, response, messages):
            if response.tool_calls:
                print_info(CodeInfo, f"触发工具调用: {response.tool_calls}")
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    tool_result = "未知工具执行失败"
                    # 动态分配工具执行
                    if tool_name == "RemenberTool":
                        tool_result = RemenberTool.invoke(tool_args)
                        print_info(CodeInfo, f"🔧 记忆工具执行结果: {tool_result}")
                    elif tool_name == "Ros2ControlTool":
                        tool_result = Ros2ControlTool.invoke(tool_args)
                        print_info(CodeInfo, f"🤖 ROS2 工具执行结果: {tool_result}")
                    elif tool_name == "VisionControlTool":
                        tool_result = VisionControlTool.invoke(tool_args)
                        print_info(CodeInfo, f"👁️ 视觉控制工具执行结果: {tool_result}")
                    elif tool_name == "PersonTrackTool":
                        tool_result = PersonTrackTool.invoke(tool_args)
                        print_info(CodeInfo, f"🚶 人物跟踪工具执行结果: {tool_result}")

                    # 将工具执行的结果封装为 ToolMessage
                    messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                        content=str(tool_result)
                    ))
                
                print_info(CodeInfo, "正在结合工具结果生成最终回复...")
                final_response = self.chat(messages)
                final_response = self.chatandspeakout(final_response)
                messages.append(final_response)
            else:
                final_response = self.chatandspeakout(response)

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
    # 根据你的 log，我帮你补充了 oh, thats, thank 等词
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

    def on_wave_detected(wave_dir, distance=None, person_3d=None):
        global last_vision_trigger_time
        current_time = time.time()
        
        # 5秒防抖冷却
        if current_time - last_vision_trigger_time < 5.0: return
        if is_speaking.is_set(): return
            
        last_vision_trigger_time = current_time
        
        # 组装带空间感知的环境提示词
        location_desc = ""
        if person_3d is not None:
            px, py, pz = person_3d
            
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
            
        info = f"{location_desc}客人正在向你挥手打招呼。"
        print_info(CodeInfo, f"[👀 收到视觉传感器反馈]: {info}")
        
        # 将带有精确坐标的视觉事件放入全局队列，LLM会在主循环中读取
        # event_queue.put({"source": "vision_system", "content": info})
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
        # event_queue.put({"source": "vision_system", "content": info})
        push_vision_to_llm(info)

    def vision_listener_worker():
        # 【修改】：将两个回调函数以及 stop_event 都传给底层
        start_gesture_recognition(
            on_wave_callback=on_wave_detected, 
            on_point_callback=on_point_detected, 
            stop_event=vision_stop_event,
            local_display=False
        )

    # 启动视觉识别守护线程
    current_vision_task = "hand"
    vision_thread=threading.Thread(target=vision_listener_worker, daemon=True)
    vision_thread.start()
    # ==================================




    # === 4. 初始化对话 ===
    system_prompt = """全程使用英语。你是一个拥有实体机器人身体的智能接待员(Embodied AI Receptionist)。
【你的职责】

    接待客人：记住客人的姓名和喜好，并在其他客人询问时进行介绍(必须调用 RemenberTool)。

    控制身体：当客人要求你执行物理动作（如：带路、拿东西、跟随）时，必须调用 Ros2ControlTool 控制你的硬件执行操作。

【重要规则】
    现在是debug环境，你只需要最简单的回答，不需要任何多余的修饰和客套话，必须依靠调用工具来完成动作。。

    在执行动作前或动作完成时，你可以用简短友好的语音回应客人（例如："I will carry the bag for you"）。

    如果你收到[SystemMessages]关于你身体或环境的状态（例如：遇到障碍物、到达目的地），你需要向客人播报当前情况。

    有时客人说的话可能会识别不准确，你需要尝试理清客人的话。如果客人的话确实是毫无意义、听不懂、只有一个单词或是乱码，不需要回答任何东西'

    所有的回答都要用于语音播报，因此请保持极其简短、口语化，不要使用 emoji 和特殊符号。
    
    当客人要求你执行move、takebag动作时，你必须调用 Ros2ControlTool 发送相应的指令给 ROS2 底层。
    例如：如果客人说“请帮我拿一下包”，你需要调用 Ros2ControlTool(name="takebag", flag="start")，并且可以说一句话回应客人，例如："I will carry the bag for you"。

    【视觉与感知能力】
    当客人要求你寻找某人、看某物、或者抓取某物时，你必须调用 VisionControlTool 开启对应的摄像头模式。
    模式包括：'face' (认人), 'object' (看东西), 'hand' (看手势), 'stop' (关闭眼睛)。

    当客人要求你跟随他的时候（follow me），你必须调用PersonTrackTool(action="start") 开启人物跟踪模式，底盘会自动跟随最近的人。当客人说“停止跟随”时，你必须调用 PersonTrackTool(action="stop") 停止跟踪。
    
    当视觉传感器传来信息时（系统提示：来自视觉传感器的反馈），你需要用自然、口语化的英语主动向客人播报。例如：
    - 看到人脸时说："Oh, hello YanZu! Nice to see you."
    - 看到物体时说："I see a cup right there."
    - 跟踪人物时说："I am following you, you are 2 meters away."
    
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="现在简单做一个开场白，只需要说一句话")
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
                    if event['content'] == "DEBUG_PING":
                        print_info(ROSInfo, "💓 [DEBUG] 收到 ROS2 底层的 10 秒存活心跳包，系统链路正常。")
                    prompt = f"【系统提示：来自底层传感器的反馈】: {event['content']}。调用Ros2ControlTool,发送给“core”程序一个“start“的flag。"
                    messages.append(HumanMessage(content=prompt))
                
                elif event["source"] == "vision_system":
                    prompt = f"【系统提示：来自视觉传感器的反馈】: {event['content']}。请用一句简短热情的话主动跟客人打招呼并询问是否需要帮助。"
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