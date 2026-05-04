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

# ===音频处理===
import pyttsx3
from scipy import signal   # 用于重采样
import edge_tts
import asyncio
import pygame
import time
from voiceV4 import FunASRSpeechTranscriber  # 这是我们改进后的 FunASRRealtime 类所在的文件
from hand_gesture.hand2 import start_gesture_recognition
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
    name: str = Field(description="机器人的程序名称")
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
        print("发送给 ROS 成功:", response.json())
    except Exception as e:
        print("发送失败，ROS 节点启动了吗？", e)
    return f"ROS2 控制指令：{name} - {flag}"

# @tool
# def CarryBagTool(item: str) -> str:
#     if item:
#         ArmCtrl = f"机械臂控制指令：请拿起{item}"
#         return ArmCtrl
    
# @tool
# def MoveToTool(location: str) -> str:
#     if location:
#         MoveCtrl = f"移动控制指令：请移动到{location}"
#         return MoveCtrl
    
# @tool
# def FollowTool(target: str) -> str:
#     if target:
#         FollowCtrl = f"跟随控制指令：请跟随{target}"
#         return FollowCtrl

# tools = [RemenberTool, CarryBagTool, MoveToTool, FollowTool]
tools = [RemenberTool, Ros2ControlTool]
is_speaking = threading.Event()

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
        self.llm_with_tools = self.client.bind_tools(tools)
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
        pygame.mixer.init()
        pygame.mixer.music.load(self.output)
        time.sleep(0.5)
        
        # 说话前，把锁拉上（捂住耳朵）
        is_speaking.set() 
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            continue
            
        # 说话完毕后，稍微等个 0.5 秒（因为录音缓冲区可能还有残留）
        time.sleep(0.5)
        # 说话结束，把锁打开（松开耳朵）
        is_speaking.clear()

    def chatandspeakout(self, messages):
        response = self.chat(messages)
        if response.content:
            print("LLM response:", response.content)
            asyncio.run(self.speakout(response.content))
            self.play_audio()
        return response

    def check_tool_calls(self, response, messages):
            if response.tool_calls:
                print(f"触发工具调用: {response.tool_calls}")
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    tool_result = "未知工具执行失败"
                    # 动态分配工具执行
                    if tool_name == "RemenberTool":
                        tool_result = RemenberTool.invoke(tool_args)
                        print(f"🔧 记忆工具执行结果: {tool_result}")
                    elif tool_name == "Ros2ControlTool":
                        tool_result = Ros2ControlTool.invoke(tool_args)
                        print(f"🤖 ROS2 工具执行结果: {tool_result}")
                        
                    # 将工具执行的结果封装为 ToolMessage
                    messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                        content=str(tool_result)
                    ))
                
                print("正在结合工具结果生成最终回复...")
                final_response = self.chatandspeakout(messages)
                messages.append(final_response)
            else:
                print("模型回复:", response.content)

# === 2. 新增：无意义字符串过滤函数 ===
def is_meaningful(text):
    text = text.strip()
    # 过滤掉太短的字符 (比如单个字母 "a", "i")
    if len(text) <= 1:
        return False
    # 过滤掉常见的 ASR 幻觉和无意义语气词 (根据你的实际情况补充)
    useless_words = ["yeah", "What", "啊", "the", "um", "uh", "hello", "hi", "test"]
    if text.lower() in useless_words:
        return False
    return True

if __name__ == "__main__":
    llm = LLMwithTools()
    voice_transcriber = FunASRSpeechTranscriber()
    
    # === 1. 创建全局事件队列 ===
    event_queue = queue.Queue()

    # === 2. 启动语音监听线程 ===
    def voice_listener_worker():
        while True:
            user_input = voice_transcriber.get_next_utterance()
            
            # 如果机器人正在说话，或者刚刚说完，直接丢弃录音结果！
            if is_speaking.is_set():
                print(f"[屏蔽回声] 机器人正在说话，已丢弃录音: {user_input}")
                continue
                
            if user_input and user_input.strip():
                # 进行无意义判断
                if not is_meaningful(user_input):
                    print(f"[过滤杂音] 识别结果无意义，已丢弃: {user_input}")
                    continue
                
                # 只有又有意义、又不是自己说的，才放进队列
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
            print(f"\n[🔔 收到 ROS2 底层反馈]: {info}")
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

    def on_wave_detected(wave_dir):
        global last_vision_trigger_time
        current_time = time.time()
        
        # 【防抖/冷却机制】：10秒内只让大模型响应一次挥手，避免死循环
        if current_time - last_vision_trigger_time < 10.0:
            return
            
        # 如果机器人正在说话，暂不打断它
        if is_speaking.is_set():
            return
            
        last_vision_trigger_time = current_time
        
        info = "客人正在向你挥手打招呼。"
        print(f"\n[👀 收到视觉传感器反馈]: {info}")
        # 将视觉事件放入全局队列
        event_queue.put({"source": "vision_system", "content": info})

    def vision_listener_worker():
        # 启动摄像头检测，并将回调函数传给它
        start_gesture_recognition(on_wave_callback=on_wave_detected)

    # 启动视觉识别守护线程
    threading.Thread(target=vision_listener_worker, daemon=True).start()
    # ==================================




    # === 4. 初始化对话 ===
    system_prompt = """全程使用英语。你是一个拥有实体机器人身体的智能接待员(Embodied AI Receptionist)。
【你的职责】

    接待客人：记住客人的姓名和喜好，并在其他客人询问时进行介绍(必须调用 RemenberTool)。

    控制身体：当客人要求你执行物理动作（如：带路、拿东西、跟随）时，必须调用 Ros2ControlTool 控制你的硬件执行操作。

【重要规则】

    在执行动作前或动作完成时，你可以用简短友好的语音回应客人（例如："I will carry the bag for you"）。

    如果你收到【系统提示】关于你身体或环境的状态（例如：遇到障碍物、到达目的地），你需要向客人播报当前情况。

    有时客人说的话可能会识别不准确，你需要尝试理清客人的话。如果客人的话毫无意义、听不懂或是乱码，请礼貌地说 'Sorry, I didn't catch that. Could you repeat?'

    所有的回答都要用于语音播报，因此请保持极其简短、口语化，不要使用 emoji 和特殊符号。
    
    现在处于debug环境，当你看到最新的prompt里有 "DEBUG_PING" 字样时，调用Ros2ControlTool,发送给“core”程序一个“start“的flag，这是告诉机器人链路正常，不需要播报给客人。

    
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="现在简单做一个开场白，只需要说一句话")
    ]
    response = llm.chatandspeakout(messages)
    messages.append(response)

# === 5. 主循环：处理事件队列 ===
    print("\n>>> 系统已就绪！可以说话，或等待 ROS2 发送事件...")
    try:
        while True:
            # 阻塞等待，直到队列里有新的事件
            event = event_queue.get() 
            
            if event["source"] == "human":
                print(f"客人说: {event['content']}")
                messages.append(HumanMessage(content=event["content"]))
                
            elif event["source"] == "robot_system":
                if event['content'] == "DEBUG_PING":
                    print("💓 [DEBUG] 收到 ROS2 底层的 10 秒存活心跳包，系统链路正常。")
                    # 继续向大模型注入指令
                prompt = f"【系统提示：来自底层传感器的反馈】: {event['content']}。调用Ros2ControlTool,发送给“core”程序一个“start“的flag。"
                messages.append(HumanMessage(content=prompt))
            
            # === 新增：处理视觉系统的事件 ===
            elif event["source"] == "vision_system":
                # 给大模型伪造一段系统提示，让它主动发话
                prompt = f"【系统提示：来自视觉传感器的反馈】: {event['content']}。请用一句简短热情的话主动跟客人打招呼并询问是否需要帮助。"
                messages.append(HumanMessage(content=prompt))
            # ===============================

            # 交给大模型处理并播报
            response = llm.chatandspeakout(messages)
            messages.append(response)
            
            # 检查大模型是否觉得需要调用工具
            llm.check_tool_calls(response, messages)
                
    except KeyboardInterrupt:
        print("系统正在退出...")
    finally:
        voice_transcriber.close()
