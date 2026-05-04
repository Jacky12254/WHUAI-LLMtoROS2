# LLM-WHURoboCup-Home

## 简介

这是**武汉大学RoboCup-Home组**的机器人核心控制工具，**使用大模型+可调用工具，以及ROS2操控机器人**。

某种程度上来说，也算是openclaw碎片。

## 代码原理

使用llama.cpp本地部署大模型，较新的大模型可以使用API连接方式；

使用Langchain中的@tool修饰紧跟的函数，作为能够被大模型调用的工具；

告知大模型工具使用方法，引导其使用；

将conda与ROS2解耦。因为其之间难以互传，本项目使用http协议，通过本地回环传递的方式进行通信，这表明通过服务器远程控制机器人是可行的；

ROS2发布话题，订阅话题并触发callback函数，将反馈信息传回大模型，并由大模型进行反应。
  

## 环境配置

Ubuntu 22.04；

conda环境如environment.yml所示；
    
ROS2可以使用鱼香ROS安装；

llama.cpp编译安装，并下载对应的大模型权重，具体开启命令可参考LLM文件夹。

不使用本地部署，使用API接口运行其他大模型，不需要llama.cpp


## 代码工作流

    语音识别/手势识别/物品识别 --->  结果传入大模型 --->  大模型发出调用工具指令(支持多个工具同时调用）并作出回答 --->  工具内部方法实现 
    
    ---http传输--->  ROS2Link节点对指令解析  --->  ROS2发布话题  --->  机器人完成操作并反馈  --->  ROS2Link节点订阅话题并传入callback函数进行解析 
    
    ---http传输--->  LLM相关代码的解析，并传入大模型 --->  大模型作出新回答与新调用 --->  语音播报/工具调用

## 代码架构

### 工具调用

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel, Field
#=======模型定义=======
arg = arg #自定义参数
@tool(args_schema = arg)
def LLMTool(arg: type) -> type: 
  '''这里写工具的方法'''
    return result

tool = [LLMTool]#这里注册工具
#=======兼容性修复=======
class PatchedChatOpenAI(ChatOpenAI):
    """
    自定义的 ChatOpenAI 类，用于修复本地模型违规返回字典格式 arguments 的兼容性问题。
    使用 *args 和 **kwargs 兼容不同版本的 LangChain 参数签名。
    """
    def _create_chat_result(self, response, *args, **kwargs):
    # 兼容对象和字典两种可能的数据格式                              
    # 将所有参数原封不动地传回给 LangChain 的原生逻辑
        return super()._create_chat_result(response, *args, **kwargs)
#=======模型初始化=======
class LLMwithTools:
    def __init__(self):
        self.client = PatchedChatOpenAI(
            base_url="http://127.0.0.1:8080",#本地端口或网络地址
            api_key="EMPTY",
            model="model_name",
            temperature=0.7,
        )
        self.llm_with_tools = self.client.bind_tools(tools, tool_choice="auto") #开启工具调用

    def checktoolcall(self,response, messages):
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]                
                tool_result = None              
                    if tool_name == "LLMTool":
                        tool_result = LLMTool.invoke(tool_args)

            messages.append(ToolMessage(
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                    content=str(tool_result)
                ))
            next_response = self.chat(messages)
            messages.append(next_response)# 生成工具使用后的回复
            # 递归调用，让大模型继续决定是否使用多个工具
            self.check_tool_calls(next_response, messages)

#=======主函数=======
if __name__="__main__":
    LLM = LLMwithTools()
    prompt = "...You have to use LLMTool(arg = arg)
    response = LLM.llm_with_tools.invoke(prompt)
    prompt.append(response)# 记录聊天记录
```

### LLM到ROS2传输

#### LLM端

```python
import threading
import queue
from flask import Flask, request, jsonify
import requests

#=======发送=======
def HTTP2ROS2(payload: dict, node: str) -> str:
    url = f"http://127.0.0.1:5000/{node}"
    response = requests.post(url, json=payload, timeout=2.0)
    return response.json()
#=======接收=======
ros_app = Flask(__name__)
    
@ros_app.route('/ros2_feedback', methods=['POST'])
def receive_ros2_feedback():
    data = request.json
    ros2_feedback_callback(data)

```

#### ROS2端

```python
app = Flask(__name__)
log.setLevel(logging.ERROR)
ros_node = None

#======发送=======
def send_feedback_to_llm(self, robot_message:str):
    '''robot_message 包括 info为反馈信息'''
    url = "http://127.0.0.1:5001/ros2_feedback"
    payload = {"info": robot_message}
    try:
        requests.post(url, json=payload, timeout=2.0)
        self.get_logger().info('已将异常状况/状态成功上报给大模型')
    except Exception as e:
        self.get_logger().error(f'llmbridge: 无法连接到大模型大脑 : {e}')

#=======接收=======
@app.route('/LLMorder', methods=['POST'])
def receive_LLMorder():
    data = request.json
    LLMorder_callback(data)

```

参考本架构，即可实现conda底下运行大模型相关，ros2节点订阅发布，两者通过http通信的效果。

## 使用方法

- clone本库
- 依据environment.yaml创建conda虚拟环境并安装packages
- 打开新终端，退出所有conda环境来开启您其他的ROS2节点
- 打开新终端/vscode, 直接运行LLMlinkRos2.py，不在虚拟环境内部即可，也可以选择先编译再运行
    ```
       colcon build
       source install/setup.bash
       ros2 run LLMlinkRos2 llmbridge
    ```
- vscode打开LLMwithTools.py,进入虚拟环境并开始运行

## 其他代码

本项目包括我为机器人准备的功能，例如语音识别，手势识别，人物追踪，大模型记忆库，语音播报和机械臂抓取等功能，可依据导入部分查看。部分代码在运行时导入路径需要绝对路径，这一部分我上传时并没有进行更正，请注意。**本项目是本地部署的Qwen3.5-9B的量化版本**。

## 贡献

项目上传人@Jacky12254,即本人，**负责本项目的架构设计，完全承担了与大模型以及大模型与Ros2通信相关的代码编写与测试。同时完全承担了语音识别，语音播报功能的编写与测试**。主要负责了人物对齐，底盘控制，手势识别等功能。合作完成了雷达导航，机械臂抓取的功能。

特别感谢其他贡献者，都是WHURobocup-Home的成员，他们在ROS2支持方面相关给力。

衷心感谢Gemini,Qwen,deepseek,CodeX等一众大模型的无私奉献。

## 免责声明

项目由多人完成，可能会上传了其他库的源代码，在此表示歉意。

本项目不得商用

## 联系方式

@Jacky12254 email: 1225423790@qq.com
    


