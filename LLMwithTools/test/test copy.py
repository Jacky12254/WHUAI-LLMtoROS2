import time
import threading
import requests
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # 关闭烦人的默认路由日志

def send_feedback_to_llm(info_text):
    """模拟底层硬件通过 HTTP 向大模型发送状态反馈"""
    url = "http://127.0.0.1:5001/ros2_feedback"
    try:
        requests.post(url, json={"info": info_text}, timeout=2.0)
        print(f"\n[Mock ROS2] 📤 已向大模型发送底层反馈: \n{info_text}\n")
    except Exception as e:
        print(f"\n[Mock ROS2] ❌ 发送反馈失败，大模型节点启动了吗？错误: {e}\n")

def simulate_hardware_action(task_name):
    """模拟底盘或机械臂的物理运动延迟，并组装特定的反馈文案"""
    # 模拟物理运动，等待 3 秒
    print(f"[Mock ROS2] ⏳ 正在模拟执行物理动作 [{task_name}]... (耗时3秒)")
    time.sleep(3)
    
    # 根据 SOP 精准构造反馈文案
    if task_name == "goto_new_guest":
        feedback = "Successfully reached the target point."
        
    elif task_name in ["goto_bar", "goto_known_guest"]:
        feedback = (
            "Path repetition stopped. You have successfully arrived at your destination. "
            "【系统最高强制指令】：请立刻根据你的任务进度，调用相应的工具（绝不允许只说话不调工具）：\n"
            "1. 如果你是来吧台取餐的，请立刻调用 Ros2ControlTool(name='grab', flag='start')；\n"
            "2. 如果你是去客桌送餐的，请立刻调用 Ros2ControlTool(name='place', flag='start')；\n"
            "3. 如果你刚完成了这一单所有的派送，并且回到了吧台等待新客人，请你必须立刻调用 VisionControlTool(task='hand') 重新开启挥手识别！"
        )
        
    elif task_name == "grab":
        feedback = "Grab task completed successfully. 【系统强制指令】：请立刻调用 Ros2ControlTool(name='goto_known_guest', flag='start') 前往已知客人位置派送餐品！"
        
    elif task_name == "place":
        feedback = "Place task completed successfully. 【系统强制指令】：如果你已经完成了本单的最后一次派送，请立刻调用 Ros2ControlTool(name='goto_bar', flag='start') 回到吧台准备接下一单！"
        
    else:
        # stop 或 wave_detect 指令不需要下发延迟反馈
        return 
        
    send_feedback_to_llm(feedback)


@app.route('/control', methods=['POST'])
def control_robot():
    """接收大模型下发的动作指令"""
    data = request.json
    name = data.get('name')
    flag = data.get('flag')
    
    print(f"\n[Mock ROS2] 📥 收到大模型指令: {name} - {flag}")
    
    if flag == 'start':
        # 开启后台线程去模拟运动，防止阻塞当前的 HTTP 响应
        threading.Thread(target=simulate_hardware_action, args=(name,), daemon=True).start()
        
    return jsonify({"status": "success", "msg": f"指令 [{name} - {flag}] 已成功下发"}), 200

# ----------------- 占位路由（防止大模型报错） -----------------
@app.route('/person_track', methods=['POST'])
def receive_person_track():
    return jsonify({"status": "ok"}), 200

@app.route('/camera_point', methods=['POST'])
def receive_camera_point():
    return jsonify({"status": "ok"}), 200

@app.route('/cmd_twist', methods=['POST'])
def receive_twist():
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    print("=====================================================")
    print("🚀 ROS2 物理环境模拟器已启动！")
    print("📡 正在监听 5000 端口，等待大模型下发指令...")
    print("=====================================================")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)