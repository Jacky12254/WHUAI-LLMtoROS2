import json
import logging
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 设置日志，方便调试
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. 使用 Pydantic 模型定义工具（保持不变）
class GuestInfo(BaseModel):
    """客人的基本信息"""
    name: str = Field(description="客人的姓名")
    preference: str = Field(description="客人的喜好")
    current_place: list[str] = Field(description="客人当前所在位置")

@tool(args_schema=GuestInfo)
def RemenberTool(name: str, preference: str, current_place: list[str]) -> str:
    """用于记忆或查询客人信息的工具。"""
    memory = {"name": name, "preference": preference, "current_place": current_place}
    # 这里是你的记忆逻辑，现在只是打印出来
    logger.info(f"工具被调用！参数为: {memory}")
    return json.dumps(memory, ensure_ascii=False)

# 2. 自定义ChatOpenAI子类，解决两个问题
class FixedQwenChatOpenAI(ChatOpenAI):
    """
    修复工具调用参数格式，并强制启用 strict 模式。
    """

    def _convert_tool_calls(self, response: dict) -> List[Dict[str, Any]]:
        """确保 arguments 字段为字符串，并补充缺失的 id。"""
        tool_calls = []
        for tc in response.get("tool_calls", []):
            func = tc["function"]
            if isinstance(func["arguments"], dict):
                func["arguments"] = json.dumps(func["arguments"])
            if "id" not in tc or not tc["id"]:
                tc["id"] = "call_" + str(abs(hash(func["arguments"])))
            tool_calls.append(tc)
        return tool_calls

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 从 kwargs 中获取工具列表（由 bind_tools 传入）
        tools = kwargs.get("tools", [])
        if tools:
            processed_tools = []
            for tool_def in tools:
                # 如果工具定义是字典，尝试为 function 添加 strict
                if isinstance(tool_def, dict) and "function" in tool_def:
                    tool_def["function"]["strict"] = True
                processed_tools.append(tool_def)
            kwargs["tools"] = processed_tools

        # 直接调用 openai 客户端（self.client 是 openai.OpenAI 实例）
        raw_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs,
        )
        raw_response = raw_response.model_dump()

        # 修复工具调用
        processed_tool_calls = self._convert_tool_calls(
            raw_response["choices"][0]["message"]
        )
        message_dict = raw_response["choices"][0]["message"]
        message_dict["tool_calls"] = processed_tool_calls

        ai_message = AIMessage(**message_dict)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

# 3. 在 LLMwithTools 类中使用修复后的模型
class LLMwithTools:
    def __init__(self):
        self.client = FixedQwenChatOpenAI(
            base_url="http://127.0.0.1:8080",
            api_key="EMPTY",
            model="Qwen3.5-9B",
            temperature=0.7,
        )
        self.llm_with_tools = self.client.bind_tools([RemenberTool])
    
    def chat(self, messages):
        response = self.llm_with_tools.invoke(messages)
        return response

# 4. 主程序
if __name__ == "__main__":
    llm = LLMwithTools()
    messages = [
        SystemMessage(content="""你是接待客人的机器人。你**必须严格**遵守以下规则：
1. 当客人告诉你他的姓名、喜好或位置时，**立即调用 `RemenberTool` 工具**来存储信息。
2. 当有其他客人询问某位客人的喜好时，**必须调用 `RemenberTool` 工具**查询，再回答。
3. 工具调用**之前，绝对不要**进行任何假设或编造信息。"""),
        HumanMessage(content="现在简单做一个开场白")
    ]
    response = llm.chat(messages)
    # ... 其余代码与你的测试代码相同
    print(response)
    while True:
        user_input = input("用户输入：")
        messages.append(HumanMessage(content=user_input))
        response = llm.chat(messages)
        print(response)