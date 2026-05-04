import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# === ROS2 ===
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
# 定义一个结构体，用来记忆客人的姓名和喜好
# class GuestMemory:
#     def __init__(self):
#         self.name = ""
#         self.preference = ""
#         self.current_place = []

# @tool
# def RemenberTool(input: str) -> str:
#     """记忆工具"""
#     '''记忆工具，输入一个字符串，解析出客人的姓名和喜好，并存储在内存中'''
#     memory = GuestMemory()
#     data = json.loads(input)
#     memory.name = data.get("name", "")
#     memory.preference = data.get("preference", "")
#     memory.current_place = data.get("current_place", [])
#     return memory.__dict__# 将内存转换为字典返回

# tools = [{
#     "type": "function",
#     "function": {
#         "name": "RemenberTool",
#         "description": "记忆工具，输入一个字符串，解析出客人的姓名和喜好，并存储在内存中",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "input": {
#                     "type": "string",
#                     "description": "输入一个字符串，格式为json，包含客人的姓名和喜好，例如：{\"name\": \"小明\", \"preference\": \"喜欢吃火锅\", \"current_place\": [\"餐厅\"]}"
#                 }
#             },
#             "required": ["input"]
#         }
#     }
# }]

from pydantic import BaseModel, Field

class GuestInfo(BaseModel):
    name: str = Field(description="客人姓名")
    preference: str = Field(description="客人喜好")
    current_place: list[str] = Field(default_factory=list, description="当前位置列表，格式是")

@tool(args_schema=GuestInfo)
def RemenberTool(name: str, preference: str, current_place: list[str]) -> str:
    """用于记忆或查询客人信息的工具。请根据客人输入调用。"""
    # 模拟记忆逻辑
    return f"已记录：{name}喜欢{preference}，在{current_place}"

tools = [RemenberTool]

# ========================== ROS2 节点示例 ==========================
class RobotCarPubNode(Node, answer=None):
    """ROS2 节点，集成研究助手并处理对话。"""
    def __init__(self, assistant: ResearchAssistant):
        super().__init__("Car_reply_node")
        self.assistant = assistant
        self.conversation_state = None
        # 这里可以添加订阅和发布器，例如订阅语音输入，发布机器人回复等
        self.get_logger().info("Research Assistant Node initialized.")
        self.publish_ = self.create_publisher(String, "Car_reply", 10)
        self.answer = answer
    
    def answer_callback(self, user_input: str):
        """处理用户输入并发布回复。"""
        self.get_logger().info(f"User: {user_input} | Assistant: {self.answer}")
        msg = String()
        msg.data = self.answer
        self.publish_.publish(msg)

class RobotArmPubNode(Node, answer=None):
    """ROS2 节点，处理机械臂控制指令。"""
    def __init__(self):
        super().__init__("arm_control_node")
        # 这里可以添加订阅和发布器，例如订阅助手的指令输出，发布机械臂控制命令等
        self.get_logger().info("Arm Control Node initialized.")
        self.publish_ = self.create_publisher(String, "arm_control_input", 10)
    
    def Arm_control_callback(self, msg: str):
        """处理机械臂控制指令。"""
        msg = String()
        msg.data = self.answer
        self.publish_.publish(msg)
        # 这里可以添加解析指令并控制机械臂的逻辑，例如调用机械臂的 API 或发布控制消息

class RobotVoiceSubNode(Node):
    """ROS2 节点，处理语音识别结果并通知研究助手。"""
    def __init__(self, assistant: ResearchAssistant):
        super().__init__("voice_recognition_node")
        self.assistant = assistant
        self.subscription_voice = self.create_subscription(String, "voice_input", self.handle_user_input, 10)
        self.get_logger().info("Voice Recognition Node initialized.")

    def voice_callback(self, msg: String):
        """接收语音识别结果并调用助手处理。"""
        user_input = msg.data
        self.get_logger().info(f"Received voice input: {user_input}")
        self.assistant.run(user_input, current_guest=self.assistant.kb.query("current_guest", "姓名"), state=self.assistant.conversation_state)

class RobotVisionSubNode(Node):
    """ROS2 节点，处理视觉识别结果并通知研究助手。"""
    def __init__(self, assistant: ResearchAssistant):
        super().__init__("vision_recognition_node")
        self.assistant = assistant
        self.subscription_vision = self.create_subscription(String, "vision_input", self.handle_vision_input, 10)
        self.get_logger().info("Vision Recognition Node initialized.")

    def vision_callback(self, msg: String):
        """接收视觉识别结果并更新助手的当前客人信息。"""
        vision_result = msg.data  # 假设格式为 "guest_name,confidence"
        guest_name, confidence_str = vision_result.split(",")
        confidence = float(confidence_str)
        self.get_logger().info(f"Received vision input: Guest: {guest_name}, Confidence: {confidence}")
        if confidence > 0.7:
            self.assistant.conversation_state["current_guest"] = guest_name
            self.get_logger().info(f"Updated current guest to: {guest_name}")
        else:
            self.get_logger().info("Vision recognition confidence too low, not updating current guest.")




class LLMwithTools:
    def __init__(self):
        """启动llm和工具"""
        self.client = ChatOpenAI(
            base_url="http://127.0.0.1:8080",
            api_key="EMPTY",
            model="Qwen3.5-9B",
            temperature=0.7,
        )
        self.llm_with_tools = self.client.bind_tools(tools)
    
    def chat(self, messages):
        """聊天接口，输入消息列表，返回回复"""
        response = self.llm_with_tools.invoke(messages)
        return response
    
    def history(self):
        """历史记录接口，返回历史消息列表"""
        return self.llm_with_tools.history


if __name__ == "__main__":
    llm = LLMwithTools()
    messages = [
        SystemMessage(content="你是个接待客人的机器人,你需要记住客人的姓名和喜好，并且向其他客人介绍这个客人。例如：客人A告诉你他的名字叫小明，喜欢吃火锅，那么你需要记住这个信息，并且当有其他客人问你小明喜欢吃什么的时候，你需要告诉他小明喜欢吃火锅。当你了解到这个客人的信息时，调用RemenberTool记忆。当被问及信息时，调用RemenberTool查询并告诉其他客人。"),
        HumanMessage(content="现在简单做一个开场白")
    ]
    response = llm.chat(messages)
    print(response)
    while True:
        user_input = input("用户输入：")
        messages.append(HumanMessage(content=user_input))
        response = llm.chat(messages)
        print(response)