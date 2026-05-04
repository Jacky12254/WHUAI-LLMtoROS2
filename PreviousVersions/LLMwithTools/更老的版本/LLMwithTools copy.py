import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
    # current_place: list[str] = Field(description="当前位置列表，格式是")

@tool(args_schema=GuestInfo)
def RemenberTool(name: str, preference: str, current_place: list[str]) -> str:
    """用于记忆或查询客人信息的工具。请根据客人输入调用。"""
    # 模拟记忆逻辑
    return f"已记录：{name}喜欢{preference}，在{current_place}"

tools = [RemenberTool]

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