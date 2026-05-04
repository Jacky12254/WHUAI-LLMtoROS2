#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from flask import Flask, request, jsonify
import threading
import requests
import random
import logging
import time
import math

from std_srvs.srv import SetBool
from std_msgs.msg import Bool

# === 1. 初始化 Flask 应用 ===
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

ros_node = None 

# === 2. 定义 ROS 2 节点 ===
class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_bridge')
        
        # 服务类型
        self.grab_client = self.create_client(SetBool, '/grab_service')
        # 发布类型
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # 订阅类型
        self.create_subscription(Bool, '/Radar', self.radar_callback, 10)
        self.RadarDataHistory = []
        
        self.current_task = None
        self.is_moving = False

        # === 人物跟踪/跟随相关状态 ===
        self.is_tracking = False           # 是否正在跟随人物
        self.follow_target_distance = 0.8    # 目标保持距离（米）
        self.follow_linear_speed = 0.3       # 线速度 m/s
        self.follow_angular_gain = 0.8       # 角速度增益
        self.last_track_data_time = 0.0      # 最后一次收到跟踪数据的时间
        self.track_data_timeout = 0.5        # 超时（秒），超过此时间未收到数据则停止
        
        # 定时器：检查跟踪数据超时
        self.create_timer(0.1, self.check_track_timeout)

        self.get_logger().info('llmbridge: ROS 2 桥接节点已完全启动！')
        self.get_logger().info('llmbridge: 正在 5000 端口监听大模型指令...')

# ============================== 处理大模型指令和反馈 ==================================
    def execute_command(self, name, flag):
        """处理来自大模型的控制指令"""
        msg = Twist()
        self.current_task = name
        
        if flag.lower() == 'start':
            if name == 'move':
                self.get_logger().info('llmbridge: 👉 收到指令：开始移动')
                msg.linear.x = 0.5
                self.is_moving = True
            elif name == 'takebag':
                self.get_logger().info('llmbridge: 👉 收到指令：开始抓取袋子')
                # 发送 True 启动抓取
                self.trigger_grab_bag(True)
                return  
            elif name == 'track':
                # 启动人物跟随模式
                self.get_logger().info('llmbridge: 👉 收到指令：开始人物跟随')
                self.is_tracking = True
                self.is_moving = True
                return  # 不发布速度，等待 /person_track 传入坐标
            elif name == 'person_follow_stop':
                # 停止人物跟随模式
                self.get_logger().info('llmbridge: 🛑 收到指令：停止人物跟随')
                self.is_tracking = False
                self.is_moving = False
                # 发送零速度停车
                self.cmd_publisher.publish(Twist())
                return
            else:
                self.get_logger().info(f'llmbridge: 👉 收到未知任务指令: {name}，默认微速前进')
                msg.linear.x = 0.1
                self.is_moving = True
                
        elif flag.lower() == 'stop':
            self.get_logger().info(f'llmbridge: 🛑 收到指令：停止 {name}')
            self.is_moving = False
            self.is_tracking = False
            
            # 如果是要求停止抓取，发送 False
            if name == 'takebag':
                self.trigger_grab_bag(False)
                return
            
        self.cmd_publisher.publish(msg)

    def send_feedback_to_llm(self, message):
        url = "http://127.0.0.1:5001/ros2_feedback"
        payload = {"info": message}
        try:
            requests.post(url, json=payload, timeout=2.0)
            if message != "DEBUG_PING":
                self.get_logger().info('llmbridge: 已将异常状况/状态成功上报给大模型大脑')
        except Exception as e:
            if message != "DEBUG_PING":
                self.get_logger().error(f'llmbridge: 无法连接到大模型大脑 : {e}')

# ============================== 处理雷达反馈 ==================================
    def radar_callback(self, msg):
        self.RadarDataHistory.append(msg.data)
        """处理来自雷达的反馈信息"""
        if msg.data:  # 假设 msg.data 为 True 表示检测到障碍物
            threading.Thread(
                target=self.send_feedback_to_llm, 
                args=("路线前方有障碍物，系统自动避障，不需要操作",)
            ).start()
            self.get_logger().warn('llmbridge: ⚠️ 雷达检测到障碍物，已触发紧急停止！')
        elif msg.data == False and len(self.RadarDataHistory) >= 2 and self.RadarDataHistory[-2] == True:
            # 从有障碍物变为无障碍物
            threading.Thread(
                target=self.send_feedback_to_llm,
                args=("前方障碍物已清除，继续执行任务",)
            ).start()
            self.get_logger().info('llmbridge: ✅ 雷达反馈：前方安全，继续执行任务。')
        else:
            self.get_logger().info('llmbridge: 雷达反馈：当前路径安全。')

# ============================== 处理人物跟踪数据 ==================================
    def process_person_track_data(self, data):
        """处理来自人物跟踪的数据并控制底盘"""
        if not self.is_tracking:
            return
        
        self.last_track_data_time = time.time()
        
        # RealSense坐标：x 正方向在右侧，z 正方向在正前方
        x = data.get('x', 0.0) 
        z = data.get('z', 0.0)
        track_id = data.get('track_id', -1)
        
        depth = abs(z)
        horizontal_offset = x
        
        # 计算角度误差：正数代表目标在右边
        angle_to_target = math.atan2(horizontal_offset, depth)
        
        msg = Twist()
        
        # --- 深度控制 (前进/后退) ---
        distance_error = depth - self.follow_target_distance
        if abs(distance_error) > 0.15:  # 扩大死区到 15cm，防止疯狂前后抖动
            # 如果太远就前进(0.3)，太近就后退(-0.15)
            msg.linear.x = self.follow_linear_speed * (1.0 if distance_error > 0 else -0.5)
        else:
            msg.linear.x = 0.0  
        
        # --- 角度控制 (转向) ---
        # ⚠️ 关键修正：ROS 中 angular.z 是逆时针为正（左转为正）。
        # 如果目标在右侧（angle_to_target > 0），我们需要右转，所以要发负的 angular.z。
        if abs(angle_to_target) > 0.08:  # 约 4.5 度的死区
            msg.angular.z = -self.follow_angular_gain * angle_to_target 
        else:
            msg.angular.z = 0.0
        
        self.cmd_publisher.publish(msg)
        
        if int(time.time()) % 3 == 0:
            self.get_logger().info(
                f'🎯 正在跟随 TrackID={track_id} | 深度={depth:.2f}m | 速度: v={msg.linear.x:.2f}, w={msg.angular.z:.2f}'
            )
     

    def check_track_timeout(self):
        """定时检查跟踪数据是否超时，超时则停车。"""
        if not self.is_tracking:
            return
        
        if time.time() - self.last_track_data_time > self.track_data_timeout:
            self.get_logger().warn('llmbridge: ⚠️ 人物跟踪数据超时，自动停车')
            self.cmd_publisher.publish(Twist())  # 停车

# ============================== 抓取相关服务调用 ==================================
    def trigger_grab_bag(self, enable=True):
        """调用抓取服务"""
        # 检查服务是否在线（等待1秒）
        if not self.grab_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('llmbridge: 抓取服务 (/grab_service) 不在线，无法执行命令！')
            return

        req = SetBool.Request()
        req.data = enable
        
        # 异步调用服务，防止阻塞 Flask 和 ROS主线程
        future = self.grab_client.call_async(req)
        # 绑定回调函数，等底层回复后再执行
        future.add_done_callback(self.grab_response_callback)
        
        action = "启动" if enable else "停止"
        self.get_logger().info(f'📤 已发送【{action}】请求给抓取节点，等待回复...')

    def grab_response_callback(self, future):
        """处理抓取服务返回的结果"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'llmbridge: 抓取节点反馈: 成功! ({response.message})')
                # 任务成功，可以上报给大模型
                threading.Thread(
                    target=self.send_feedback_to_llm, 
                    args=("Grab task completed successfully.",)
                ).start()
            else:
                self.get_logger().warn(f'llmbridge: 抓取节点反馈: 失败! ({response.message})')
                # 任务失败，上报给大模型求助或记录
                threading.Thread(
                    target=self.send_feedback_to_llm, 
                    args=(f"Grab task failed. Reason: {response.message}",)
                ).start()
        except Exception as e:
            self.get_logger().error(f'llmbridge: 服务调用过程发生异常: {e}')


# ======================================= 定义 Flask 路由 =============================================
@app.route('/control', methods=['POST'])
def control_robot():
    data = request.json
    name = data.get('name')
    flag = data.get('flag')
    
    if ros_node and name and flag:
        ros_node.execute_command(name, flag)
        return jsonify({
            "status": "success", 
            "msg": f"指令 [{name} - {flag}] 已成功下发至 ROS 2 底层"
        }), 200
    else:
        return jsonify({"status": "error", "msg": "缺少必要的参数 name 或 flag"}), 400

# === 新增：人物跟踪数据接收路由 ===
@app.route('/person_track', methods=['POST'])
def receive_person_track():
    """
    接收来自 conda 环境（PersonTrackingServer）的人物跟踪数据。
    
    请求体 JSON:
    {
        "track_id": 1,
        "x": 0.15,     # 水平偏移（米），左负右正
        "y": -0.05,    # 垂直偏移（米）
        "z": 1.8,      # 深度距离（米）
        "is_tracking": true
    }
    """
    data = request.json
    if not data or not ros_node:
        return jsonify({"status": "error", "msg": "无效数据或节点未就绪"}), 400
    
    is_tracking = data.get('is_tracking', False)
    if not is_tracking:
        return jsonify({"status": "ok", "msg": "未在跟踪状态"}), 200
    
    # 交给 ROS2 节点处理
    ros_node.process_person_track_data(data)
    return jsonify({"status": "ok"}), 200

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# === 4. 主程序入口 ===
def main(args=None):
    global ros_node
    rclpy.init(args=args)
    ros_node = RobotControlNode()
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        ros_node.get_logger().info('收到退出信号，正在关闭节点...')
    finally:
        if ros_node:
            ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
