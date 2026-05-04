# 文件名：ros2_http_bridge.py (原生环境运行)
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from flask import Flask, request, jsonify
import threading
import requests

app = Flask(__name__)
# 全局变量，用于存放 ROS 节点实例
ros_node = None 

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_bridge')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('ROS 2 桥接节点已启动，等待 HTTP 指令...')

    def execute_command(self, action):
        msg = Twist()
        if action == 'forward':
            msg.linear.x = 0.5
        elif action == 'stop':
            msg.linear.x = 0.0
        self.publisher_.publish(msg)
        self.get_logger().info(f'已执行指令: {action}')

# Flask 路由：接收来自 Conda 的 POST 请求
@app.route('/control', methods=['POST'])
def control_robot():
    data = request.json
    action = data.get('action')
    if ros_node and action:
        ros_node.execute_command(action)
        return jsonify({"status": "success", "msg": f"指令 {action} 已下发至 ROS"}), 200
    return jsonify({"status": "error"}), 400

def run_flask():
    app.run(host='127.0.0.1', port=5000)


def send_feedback_to_llm(message):
    url = "http://127.0.0.1:5001/ros2_feedback"
    payload = {"info": message}
    try:
        # 给大模型发通知
        requests.post(url, json=payload, timeout=1) 
    except Exception as e:
        print("大模型端未启动", e)

# 举例：在 ROS2 节点中调用
# 当雷达检测到障碍物时：
# send_feedback_to_llm("There is an obstacle ahead, I had to stop.")
def main():
    global ros_node
    rclpy.init()
    ros_node = RobotControlNode()
    
    # 开启一个新线程专门跑 Flask，不阻塞 ROS 的 spin
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # 主线程跑 ROS 节点
    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()