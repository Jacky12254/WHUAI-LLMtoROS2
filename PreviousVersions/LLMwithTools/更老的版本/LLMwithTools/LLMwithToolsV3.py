import os
import json
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel, Field
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
        response = requests.post(url, json=payload)
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
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue

    def chatandspeakout(self, messages):
        response = self.chat(messages)
        if response.content:
            print("LLM response:", response.content)
            asyncio.run(self.speakout(response.content))
            self.play_audio()
        return response

    def check_tool_calls(self,response,messages):
        if response.tool_calls:
            print(f"触发工具调用: {response.tool_calls}")
            
            # 遍历并执行所有被调用的工具
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if tool_name == "RemenberTool":
                    # 真实执行你的工具函数
                    tool_result = RemenberTool.invoke(tool_args)
                    print(f"🔧 工具执行结果: {tool_result}")
                    
                    # 4. 【关键步骤】将工具执行的结果封装为 ToolMessage 塞进历史记录
                    messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                        content=str(tool_result)
                    ))
            
            # 5. 工具执行完毕并反馈给模型后，再次调用模型，让它结合记忆结果给用户一句话答复
            print("正在结合工具结果生成最终回复...")
            final_response = self.chatandspeakout(messages)
            messages.append(final_response)    # 拿到结果后构造 ToolMessage 添加到 messages，再 invoke 一次模型让它回答。
        else:
            print("模型回复:", response.content)


if __name__ == "__main__":
    llm = LLMwithTools()
    voice_transcriber = FunASRSpeechTranscriber()
    messages = [
        SystemMessage(content="全程使用英语。你是个接待客人的机器人,你需要记住客人的姓名和喜好，并且向其他客人介绍这个客人。例如：客人A告诉你他的名字叫小明，喜欢吃火锅，那么你需要记住这个信息，并且当有其他客人问你小明喜欢吃什么的时候，你需要告诉他小明喜欢吃火锅。当你了解到这个客人的信息时，必须调用RemenberTool记忆。当被问及信息时，必须调用RemenberTool查询并告诉其他客人。注意：客人使用语音系统来向你表达信息，所以你接受信息后要先自己梳理客人的信息，减少错误发生。你所回答的话都是要语音播报的，简单回答，不要emoji，也不要有特殊符号。"),
        HumanMessage(content="现在简单做一个开场白，只需要说一句话")
    ]
    response = llm.chatandspeakout(messages)

    try:
        while True:
            print("\n>>> 正在倾听客人说话... (请对着麦克风说)")
            
            # 3. 替代原有的 input()，使用 Vosk 阻塞监听，直到客人说完一整句话
            user_input = voice_transcriber.get_next_utterance()
            print(f"客人说: {user_input}")
            
            # 如果识别到了文字，再传给大模型
            if user_input.strip():
                messages.append(HumanMessage(content=user_input))
                response = llm.chatandspeakout(messages)
                messages.append(response) # 把 AI 的回复加入历史
                
                # 检查是否触发了工具调用 (记忆或查询)
                llm.check_tool_calls(response, messages)
                
    except KeyboardInterrupt:
        print("系统正在退出...")
    finally:
        # 清理麦克风资源
        voice_transcriber.close()
