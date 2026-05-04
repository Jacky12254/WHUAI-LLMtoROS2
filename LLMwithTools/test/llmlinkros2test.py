import requests
import time
import threading
from flask import Flask, request, jsonify

# === 1. 模拟启动大模型端的反馈接收服务器 (监听 5001 端口) ===
app = Flask(__name__)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/ros2_feedback', methods=['POST'])
def receive_ros2_feedback():
    data = request.json
    info = data.get('info', '')
    print(f"\n[🔔 收到 ROS2 底层反馈]: {info}")
    return jsonify({"status": "received"}), 200

def run_feedback_server():
    print(">>> 正在启动反馈监听服务器 (Port: 5001)...")
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

threading.Thread(target=run_feedback_server, daemon=True).start()
time.sleep(1) # 等待服务器启动

# === 2. 模拟 LLM 工具调用，向 LLMlinkRos2 (5000端口) 发送控制指令 ===
ROS2_CONTROL_URL = "http://127.0.0.1:5000/control"
ROS2_TRACK_URL = "http://127.0.0.1:5000/person_track"

def send_control(name, flag):
    payload = {"name": name, "flag": flag}
    try:
        response = requests.post(ROS2_CONTROL_URL, json=payload, timeout=2.0)
        print(f"[发出控制指令] {name} - {flag} | 状态: {response.json()}")
    except Exception as e:
        print(f"[通讯失败] 无法连接到 ROS 2 控制桥接节点: {e}")

def send_track_data(track_id, x, z, is_tracking=True):
    payload = {
        "track_id": track_id,
        "x": x,
        "y": 0.0,
        "z": z,
        "is_tracking": is_tracking
    }
    try:
        requests.post(ROS2_TRACK_URL, json=payload, timeout=1.0)
        print(f"[发出追踪数据] 目标ID:{track_id}, 坐标(x:{x}, z:{z})")
    except Exception as e:
        print(f"[通讯失败] 追踪数据发送失败: {e}")

# === 3. 执行测试序列 ===
if __name__ == "__main__":
    print("\n--- 开始测试 LLMlinkRos2 通讯 ---")
    
    # 测试 1：常规移动指令
    print("\n>>> 1. 测试常规移动指令 (move - start)")
    send_control("move", "start")
    time.sleep(2)
    send_control("move", "stop")
    time.sleep(1)

    # 测试 2：服务调用指令 (抓取)
    print("\n>>> 2. 测试服务调用 (takebag - start)")
    send_control("takebag", "start")
    time.sleep(2) # 留出时间观察 5001 端口是否收到成功/失败反馈

    # 测试 3：人物追踪连续数据流
    print("\n>>> 3. 测试人物跟随 (track - start & 发送坐标)")
    send_control("track", "start")
    time.sleep(1)
    
    # 模拟发送 5 帧视觉追踪数据
    for i in range(5):
        # 模拟人物在右前方（x=0.2），距离由 1.5 米靠近到 1.0 米
        send_track_data(track_id=1, x=0.2, z=1.5 - (i * 0.1))
        time.sleep(0.3) # 模拟摄像头帧率
        
    print("\n>>> 4. 测试人物追踪超时停车机制 (停止发送数据)")
    time.sleep(1.5) # LLMlinkRos2 中设置了 0.5 秒超时，这里等待 1.5 秒观察是否触发
    
    print("\n>>> 5. 发送停止跟随指令")
    send_control("person_follow_stop", "stop")
    
    print("\n--- 测试完成，等待 3 秒观察最后的反馈 ---")
    time.sleep(3)