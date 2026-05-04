# LLM-WHURoboCup-Home

## 简介

这是武汉大学RoboCup-Home组的机器人核心控制工具，使用大模型+可调用工具，以及ROS2操控机器人。

某种程度上来说，也算是openclaw碎片。

## 代码原理

使用llama.cpp本地部署大模型，较新的大模型可以使用API连接方式；

使用Langchain中的@tool修饰紧跟的函数，作为能够被大模型调用的工具；

告知大模型工具使用方法，引导其使用；

由于conda与ROS2之间难以互传，本项目使用http协议，通过本地回环传递的方式进行通信，这表明通过服务器远程控制机器人是可行的；

ROS2发布话题，订阅话题并触发callback函数，将反馈信息传回大模型，并由大模型进行反应。
  

## 环境配置

Ubuntu 22.04；

conda环境如environment.yaml所示；
    
ROS2可以使用鱼香ROS安装；

llama.cpp安装方式如下：

    

## 代码工作流

    语音识别/手势识别/物品识别 --->  结果传入大模型 --->  大模型发出调用工具指令(支持多个工具同时调用）并作出回答 --->  工具内部方法实现 
    
    ---http传输--->  ROS2Link节点对指令解析  --->  ROS2发布话题  --->  机器人完成操作并反馈  --->  ROS2Link节点订阅话题并传入callback函数进行解析 
    
    ---http传输--->  LLM相关代码的解析，并传入大模型 --->  大模型作出新回答与新调用 --->  语音播报/工具调用

## 代码架构

### 工具调用

```python

    import Langchain
    @tool(arg = arg)
    def LLMTool(arg: type)-> type
        '''这里写工具的方法'''
        return

    tool = [LLMTool]#这里注册工具

### LLM到ROS2传输
```python

    
    


