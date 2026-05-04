#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from flask import Flask, request, jsonify
import threading
import requests
import random
import logging

# === 1. 初始化 Flask 应用 ===
app = Flask(__name__)
# 关掉 Flask 默认的烦人输出，保持终端清爽
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# 全局变量，用于在 Flask 路由中访问 ROS 节点实例
ros_node = None 

# === 2. 定义 ROS 2 节点 ===
class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_bridge')
        
        # 创建一个发布者，用于控制机器人的移动 (发布到 /cmd_vel)
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 定时器 1：模拟底层硬件周期性检查（例如：雷达避障，2秒检查一次）
        self.timer = self.create_timer(2.0, self.simulate_hardware_check)
        
        # 定时器 2：发送调试心跳包（10秒发送一次）
        self.debug_timer = self.create_timer(10.0, self.send_debug_ping)
        
        # 内部状态变量
        self.current_task = None
        self.is_moving = False

        self.get_logger().info('🚀 ROS 2 桥接节点已完全启动！')
        self.get_logger().info('📡 正在 5000 端口监听大模型指令...')

    def execute_command(self, name, flag):
        """处理来自大模型的控制指令"""
        msg = Twist()
        self.current_task = name
        
        if flag.lower() == 'start':
            if name == 'follow':
                self.get_logger().info('👉 收到指令：开始跟随模式')
                msg.linear.x = 0.3 
                self.is_moving = True
            elif name == 'move':
                self.get_logger().info('👉 收到指令：开始移动')
                msg.linear.x = 0.5
                self.is_moving = True
            else:
                self.get_logger().info(f'👉 收到未知任务指令: {name}，默认微速前进')
                msg.linear.x = 0.1
                self.is_moving = True
                
        elif flag.lower() == 'stop':
            self.get_logger().info(f'🛑 收到指令：停止 {name}')
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.is_moving = False
            
        # 发布控制消息给底层硬件
        self.cmd_publisher.publish(msg)

    def simulate_hardware_check(self):
        """模拟硬件传感器反馈（例如遇到障碍物）"""
        # 如果机器人正在移动，有 10% 的概率假装遇到障碍物
        if self.is_moving and random.random() < 0.1:
            self.get_logger().warn('⚠️ 模拟器：前方发现障碍物，紧急停车并上报大模型！')
            
            # 1. 物理层先紧急停车保命
            stop_msg = Twist()
            self.cmd_publisher.publish(stop_msg)
            self.is_moving = False
            
            # 2. 通过 HTTP 异步通知大模型所在的 5001 端口
            threading.Thread(
                target=self.send_feedback_to_llm, 
                args=("Danger! An obstacle suddenly appeared in front of me, I have triggered emergency stop.",)
            ).start()

    def send_debug_ping(self):
        """每隔 10 秒向大模型发送一次存活调试信号"""
        threading.Thread(target=self.send_feedback_to_llm, args=("DEBUG_PING",)).start()

    def send_feedback_to_llm(self, message):
        """将底层事件或心跳发送给大模型的事件队列"""
        url = "http://127.0.0.1:5001/ros2_feedback"
        payload = {"info": message}
        try:
            # 必须加 timeout，防止大模型没启动导致 ROS 线程卡死
            requests.post(url, json=payload, timeout=2.0)
            if message != "DEBUG_PING":
                self.get_logger().info('✅ 已将异常状况成功上报给大模型大脑')
        except Exception as e:
            if message != "DEBUG_PING":
                self.get_logger().error(f'❌ 无法连接到大模型大脑 (端口 5001 未开启?) : {e}')


# === 3. 定义 Flask 路由 (接收大模型指令) ===
@app.route('/control', methods=['POST'])
def control_robot():
    """接收大模型调用 Ros2ControlTool 发来的 JSON"""
    data = request.json
    name = data.get('name')
    flag = data.get('flag')
    
    if ros_node and name and flag:
        # 在 ROS 节点中执行相应的动作
        ros_node.execute_command(name, flag)
        return jsonify({
            "status": "success", 
            "msg": f"指令 [{name} - {flag}] 已成功下发至 ROS 2 底层"
        }), 200
    else:
        return jsonify({"status": "error", "msg": "缺少必要的参数 name 或 flag"}), 400

def run_flask():
    """在独立线程中运行 Flask"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


# === 4. 主程序入口 ===
def main(args=None):
    global ros_node
    
    # 初始化 ROS 2 客户端库
    rclpy.init(args=args)
    
    # 实例化节点
    ros_node = RobotControlNode()
    
    # 开启一个后台守护线程运行 Flask 服务器，不阻塞 ROS 的 spin()
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    try:
        # 保持 ROS 节点持续运行，处理定时器和订阅回调
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        ros_node.get_logger().info('收到退出信号，正在关闭节点...')
    finally:
        # 清理工作
        if ros_node:
            ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()