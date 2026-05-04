# === Standard Library ===
import os
from typing import Literal, Optional

# === Data Modeling ===
from pydantic import BaseModel, Field

# === LangChain / Tools ===
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.outputs import ChatResult
# === LangGraph ===
from langgraph.graph import END, START, StateGraph, MessagesState

# === ChromaDB / Embeddings ===
import chromadb
from chromadb.utils import embedding_functions

# === ROS2 ===
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json

# ========================== 知识库模块 ==========================
class GuestKnowledgeBase:
    """
    客人信息知识库（仅文本），使用 ChromaDB 存储，
    以客人姓名作为元数据过滤字段。
    """
    def __init__(self, chroma_path="./chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="/home/jacky/qwen3/bge-small-zh-v1.5"  # 离线轻量中文嵌入模型
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="guest_info",
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )


    def query(self, guest_name: str, question: str, top_k: int = 3) -> str:
        """
        根据客人姓名和问题检索相关信息。
        :param guest_name: 客人姓名（由视觉模型提供）
        :param question:   查询内容，如“爱好”
        :param top_k:      返回最相关的片段数
        :return:           拼接的文本片段
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            where={"guest_name": guest_name}  # 直接使用姓名过滤
        )
        docs = results['documents'][0] if results['documents'] else []
        if not docs:
            return f"没有找到关于 {guest_name} 的 {question} 相关信息。"
        return "\n".join(docs)

    def add_guest_info(self, guest_name: str, hobbies: str):
        """
        添加新客人的文本信息（录入时调用）
        :param guest_name: 客人姓名
        :param hobbies:    爱好描述（可包含多条，用分号分隔）
        """
        # 添加姓名（便于检索）
        self.collection.add(
            documents=[f"姓名：{guest_name}"],
            metadatas=[{"guest_name": guest_name, "field": "name"}],
            ids=[f"{guest_name}_name"]
        )
        # 添加爱好（作为一条文档，也可拆分为多条）
        self.collection.add(
            documents=[f"爱好：{hobbies}"],
            metadatas=[{"guest_name": guest_name, "field": "hobby"}],
            ids=[f"{guest_name}_hobby"]
        )

    def close(self):
        """ChromaDB 客户端无需显式关闭，但可保留方法以便扩展"""
        pass


# ========================== 助手状态定义 ==========================
class ScratchpadState(MessagesState):
    """Agent state with scratchpad and current guest info."""
    scratchpad: str = Field(default="", description="The scratchpad for storing notes")
    current_guest: Optional[str] = Field(default=None, description="Name of the current guest identified by vision model")

# ========================== 兼容 OpenAI 协议的 ChatOpenAI 包装类 ==========================
class CompatibleChatOpenAI(ChatOpenAI):
    def _create_chat_result(self, response: any, *args: any, **kwargs: any) -> ChatResult:
        # 1. 统一转换为字典格式处理
        if hasattr(response, "model_dump"):
            # 如果是 Pydantic v2 对象 (新版 OpenAI SDK)
            response_dict = response.model_dump()
        elif hasattr(response, "dict"):
            # 如果是 Pydantic v1 对象
            response_dict = response.dict()
        elif not isinstance(response, dict):
            # 最后的保底手段：尝试强转
            try:
                response_dict = dict(response)
            except:
                response_dict = response
        else:
            response_dict = response

        # 2. 修改字典中的数据
        if isinstance(response_dict, dict) and "choices" in response_dict:
            for choice in response_dict.get("choices", []):
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        # 重点：修正 local LLM 返回的 dict 格式参数
                        function_info = tc.get("function", {})
                        if isinstance(function_info.get("arguments"), dict):
                            function_info["arguments"] = json.dumps(
                                function_info["arguments"], 
                                ensure_ascii=False
                            )

        # 3. 将修改后的字典传给父类
        # LangChain 的 ChatOpenAI._create_chat_result 内部会自动处理字典或对象
        return super()._create_chat_result(response_dict, *args, **kwargs)

# ========================== 研究助手类 ==========================
class ResearchAssistant:
    """
    具备 scratchpad、网络搜索和客人知识库查询功能的研究助手。
    使用兼容 OpenAI 协议的大模型接口（如 llama.cpp 提供的 API）。
    """
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 1.0,
        tavily_api_key: Optional[str] = None,
        kb: Optional[GuestKnowledgeBase] = None
    ):
        """
        初始化助手。
        :param base_url: OpenAI 兼容 API 的基础 URL，例如 "http://127.0.0.1:8001/v1"
        :param api_key: API 密钥，可设置为 "sk-no-key-required"
        :param model_name: 模型名称，例如 "unsloth/Qwen3.5-397B-A17B"
        :param temperature: 温度参数
        :param tavily_api_key: Tavily 搜索 API 密钥，若为 None 则从环境变量读取
        :param kb: 客人知识库实例，若为 None 则自动创建默认实例
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")

        # 初始化知识库
        self.kb = kb if kb is not None else GuestKnowledgeBase()

        self._setup_tools()
        self._setup_llm()
        self._build_graph()

    def _setup_tools(self):
        """定义所有可用工具。"""
        # ---------- Scratchpad 工具 ----------
        class WriteToScratchpad(BaseModel):
            """Tool to write notes into the scratchpad memory."""
            notes: str = Field(description="Notes to save to the scratchpad")

        class ReadFromScratchpad(BaseModel):
            """Tool to read previously saved notes from the scratchpad."""
            reasoning: str = Field(description="Why the agent wants to retrieve past notes")

        self.write_tool = tool(
            args_schema=WriteToScratchpad,
            return_direct=False,
            description="Tool to write notes to the scratchpad."
        )(lambda notes: f"Wrote to scratchpad: {notes}")

        self.read_tool = tool(
            args_schema=ReadFromScratchpad,
            return_direct=False,
            description="Tool to read reasoning from the scratchpad."
        )(lambda reasoning: "Reading from scratchpad")

        # ---------- 网络搜索工具 ----------
        if self.tavily_api_key:
            self.search_tool = TavilySearch(
                max_results=5,
                topic="general",
                api_key=self.tavily_api_key
            )
        else:
            # 如果没有 API key，创建一个哑工具
            class DummySearch(BaseModel):
                query: str = Field(description="search query")

                
            self.search_tool = tool(
                args_schema=DummySearch,
                return_direct=False,
                description="Dummy search tool that returns a message when no API key is provided."
            )(lambda query: "Search is disabled (no API key).")

        # ---------- 客人知识库查询工具 ----------
        class QueryGuest(BaseModel):
            """查询客人的个性化信息（如爱好、喜好等）"""
            guest_name: str = Field(description="客人的姓名")
            question: str = Field(description="需要查询的具体问题，例如'爱好'")

        self.query_guest_tool = tool(
            args_schema=QueryGuest,
            return_direct=False,
            description="Tool to query guest's personalized information."
        )(lambda guest_name, question: "Query guest")  # 实际逻辑在 tool_node 中处理

        # 收集所有工具
        self.tools = [
            self.read_tool,
            self.write_tool,
            self.search_tool,
            self.query_guest_tool
        ]
        self.tools_by_name = {tool.name: tool for tool in self.tools}

    def _setup_llm(self):
        """初始化兼容 OpenAI 协议的 LLM，并绑定工具。"""
        self.llm = CompatibleChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model_name,
            temperature=self.temperature,
            model_kwargs={"parallel_tool_calls": False},
            )
        # 如果模型不支持原生工具调用，可以设置 model_kwargs={"parallel_tool_calls": False}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    # # 添加消息格式转换函数（放在类内部）
    # def _convert_messages(self, messages):
    #     """
    #     将消息列表转换为仅包含纯文本 content 的新消息对象列表。
    #     如果 content 是 list，则提取所有文本部分并拼接。
    #     返回的新列表与原 messages 无引用关系。
    #     """
    #     converted = []
    #     for msg in messages:
    #         if hasattr(msg, 'content'):
    #             content = msg.content
    #             if isinstance(content, list):
    #                 # 提取文本内容
    #                 text_parts = []
    #                 for part in content:
    #                     if isinstance(part, dict) and part.get("type") == "text":
    #                         text_parts.append(part.get("text", ""))
    #                     elif isinstance(part, str):
    #                         text_parts.append(part)
    #                 content = " ".join(text_parts)
    #             # 只保留 content 字段，其他字段丢弃
    #             msg_class = type(msg)
    #             # 只传递 content，避免保留原对象的引用
    #             try:
    #                 new_msg = msg_class(content=content)
    #             except Exception:
    #                 # 如果构造失败，降级为字符串
    #                 new_msg = content
    #             converted.append(new_msg)
    #         else:
    #             # 非标准消息对象，直接转为字符串
    #             converted.append(str(msg))
    #     return converted

    def _build_graph(self):
        """构建 LangGraph 计算图。"""
        # ---------- 系统提示词 ----------
        self.scratchpad_prompt = """You are a friendly robot assistant with access to:
- Web search (TavilySearch)
- A persistent scratchpad (WriteToScratchpad / ReadFromScratchpad)
- A guest knowledge base (query_guest) that stores personalized information about guests, such as their hobbies and preferences.

**Important**: Before each conversation, you will be informed of the current guest's name (if available). Use this information to resolve pronouns like "he", "she", or "this guest" in user questions.

When you need to retrieve information about a guest, use the `query_guest` tool with the correct guest name. If the question refers to the current guest, provide the name you were given.

Available tools:
- ReadFromScratchpad
- WriteToScratchpad
- TavilySearch
- query_guest
"""

        # ---------- 节点函数 ----------
        def llm_call(state: ScratchpadState) -> dict:
            """调用 LLM 生成响应（可能包含工具调用）。"""
            # 基础消息：系统提示
            system_content = self.scratchpad_prompt
            if state.get("current_guest"):
                system_content += f"\n\n当前正在与客人 {state['current_guest']} 对话。如果用户问题中提及'他'、'她'或'这位客人'，指的就是{state['current_guest']}。"
            
            messages = [SystemMessage(content=system_content)]

            # 添加对话历史
            messages.extend(state["messages"])

            # 调用模型
            # 转换消息格式
            # messages = self._convert_messages(messages)
            
            response = self.llm_with_tools.invoke(messages)
            
            # 修复工具调用参数格式
            if hasattr(response, 'tool_calls') and response.tool_calls:
                import json
                for tc in response.tool_calls:
                    if isinstance(tc.get('args'), dict):
                        tc['args'] = json.dumps(tc['args'])
            return {"messages": [response]}

        def tool_node(state: ScratchpadState) -> dict:
            """执行所有工具调用，并更新 scratchpad 状态。"""
            last_message = state["messages"][-1]
            tool_calls = last_message.tool_calls
            results = []
            new_scratchpad = state.get("scratchpad", "")

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                if tool_name == "ReadFromScratchpad":
                    content = f"Notes from scratchpad: {new_scratchpad}"
                    results.append(ToolMessage(content=content, tool_call_id=tool_id))

                elif tool_name == "WriteToScratchpad":
                    notes = tool_args.get("notes", "")
                    new_scratchpad = notes  # 覆盖模式，可改为追加
                    content = f"Wrote to scratchpad: {notes}"
                    results.append(ToolMessage(content=content, tool_call_id=tool_id))

                elif tool_name == "tavily_search":
                    query = tool_args.get("query", "")
                    if hasattr(self.search_tool, "invoke"):
                        search_result = self.search_tool.invoke({"query": query})
                        content = str(search_result)
                    else:
                        content = "Search tool not available."
                    results.append(ToolMessage(content=content, tool_call_id=tool_id))

                elif tool_name == "query_guest":
                    guest_name = tool_args.get("guest_name")
                    question = tool_args.get("question")
                    content = self.kb.query(guest_name, question)
                    results.append(ToolMessage(content=content, tool_call_id=tool_id))

                else:
                    # 未知工具
                    results.append(ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_id))

            return {"messages": results, "scratchpad": new_scratchpad}

        def should_continue(state: ScratchpadState) -> Literal["tool_node", "__end__"]:
            """判断是否继续执行工具调用。"""
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tool_node"
            return END

        # 构建图
        builder = StateGraph(ScratchpadState)
        builder.add_node("llm_call", llm_call)
        builder.add_node("tool_node", tool_node)
        builder.add_edge(START, "llm_call")
        builder.add_conditional_edges("llm_call", should_continue, {"tool_node": "tool_node", END: END})
        builder.add_edge("tool_node", "llm_call")

        self.agent = builder.compile()

    def run(self, query: str, current_guest: Optional[str] = None, state: Optional[ScratchpadState] = None):
        """
        执行一次对话，支持多轮。
        :param query: 用户输入
        :param current_guest: 当前客人姓名
        :param state: 之前的状态（若为 None，则创建新状态）
        :return: (new_state, final_message)
        """
        if state is None:
            # 首次调用，创建新状态
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "current_guest": current_guest,
                "scratchpad": ""
            }
        else:
            # 复用已有状态，追加用户消息
            initial_state = {
                "messages": state["messages"] + [HumanMessage(content=query)],
                "current_guest": current_guest if current_guest is not None else state.get("current_guest"),
                "scratchpad": state.get("scratchpad", "")
            }
        final_state = self.agent.invoke(initial_state)
        return final_state, final_state["messages"][-1].content

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
            
# ========================== 使用示例 ==========================
def simulate_vision_recognition(times):
    """
    模拟视觉模型输出。
    实际场景中应调用摄像头和人脸识别模型。
    """
    # 假设返回 (客人姓名, 置信度)
    # 为演示，硬编码为 "张三" 或 "unknown"
    if times > 0:
        return "张三", 0.95
    else:
        return "unknown", 0.0

if __name__ == "__main__":
    # 1. 初始化助手（请根据实际情况修改 base_url 和 model_name）
    assistant = ResearchAssistant(
        base_url="http://127.0.0.1:8080/v1",   # 你的 llama.cpp 服务器地址
        api_key="sk-no-key-required",          # 随意填写
        model_name="Qwen3.5-9B",# 你的模型名称
        temperature=0.7
    )
    # 会话状态
    conversation_state = None
    times = 0
    while True:
        
        user_input = input("你: ")

            # 第一次调用时 state=None，后续传入上一次的状态
        conversation_state, answer = assistant.run(user_input, current_guest="张三", state=conversation_state)
        print("机器人:", answer)
        
        if conversation_state != None:
            guest_name, confidence = simulate_vision_recognition(times)
            print(f"[视觉识别] 客人: {guest_name}, 置信度: {confidence}")

        if times == 0:
            times += 1

            # 2. 模拟视觉识别
            
            if guest_name == "unknown" or guest_name == "" or confidence < 0.7:
                # 3a. 新客人或低置信度：启动录入流程
                print("机器人: 您好，看起来您是第一次来，能告诉我您的姓名和爱好吗？")
                # 模拟用户输入（实际应为语音识别）
                name = input("请输入您的姓名: ")
                hobby = input("请输入您的爱好（如：打篮球、读书）: ")
                # 存储到知识库

                assistant.kb.add_guest_info(name, hobby)
                print(f"机器人: 谢谢 {name}，已记住您的爱好！")
                # 可选：通知视觉模块绑定人脸特征（此处略）
        else:
            # 3b. 老客人：正常问答
            user_question = input("用户: ")  # 例如 "他喜欢什么运动？"
            messages, scratchpad = assistant.run(user_question, current_guest=guest_name, state=conversation_state)
            final_answer = messages[-1].content
            print("机器人:", final_answer)

    # 关闭知识库（如有需要）
    assistant.kb.close()