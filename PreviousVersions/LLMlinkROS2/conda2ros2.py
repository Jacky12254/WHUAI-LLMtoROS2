# 文件名：llm_brain.py (Conda 环境运行)
import requests
import time

# 假设这里是你和大模型通信的代码
def chat_with_llm(prompt):
    # 模拟大模型的思考结果：识别到需要让机器人前进
    print(f"大模型正在思考：{prompt}")
    time.sleep(1) # 模拟延迟
    return "forward" 

def send_to_ros(action):
    url = "http://127.0.0.1:5000/control"
    payload = {"action": action}
    try:
        response = requests.post(url, json=payload)
        print("发送给 ROS 成功:", response.json())
    except Exception as e:
        print("发送失败，ROS 节点启动了吗？", e)

if __name__ == "__main__":
    while True:
        user_input = input("请输入你想对机器人说的话：")
        # 1. 问大模型，拿到动作
        action = chat_with_llm(user_input)
        # 2. 通过 HTTP 发给 ROS 2
        send_to_ros(action)