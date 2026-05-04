from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, Literal
import json
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
import time
import random
import math
from typing import Dict, Any, Optional
import re

# 模型输出结构化指令 - 添加抓取相关动作
class RobotCommand(BaseModel):
    action: Literal[
        "MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", 
        "STOP", "AVOID_OBSTACLE", "RESUME_TASK",
        "GRAB_OBJECT", "RELEASE_OBJECT", "MOVE_ARM_UP", "MOVE_ARM_DOWN",
        "SCAN_FOR_OBJECTS", "APPROACH_OBJECT"
    ]
    speed: Optional[float] = 0.5
    duration: Optional[float] = 1.0
    target_object: Optional[str] = None  # 目标物体
    arm_position: Optional[float] = 0.0  # 机械臂位置
    reason: Optional[str] = None

# 机器人状态
class RobotState(BaseModel):
    sensor_data: dict = {}
    task_status: str = "IDLE"
    last_command: Optional[RobotCommand] = None
    error: Optional[str] = None
    inventory: list = []  # 抓取到的物体
    arm_angle: float = 0.0  # 机械臂角度

class MemoryandRead:
    def __init__(self, core_prompt: str, initial_task: str = ""):
        self.store = InMemoryStore()
        self.namespace_core = ("robot", "core")
        self.namespace_target = ("robot", "target")
        self.namespace_inventory = ("robot", "inventory")
        
        # 存储核心提示词
        self.store.put(self.namespace_core, "prompt", {"content": core_prompt})
        
        # 存储初始任务
        if initial_task:
            self.store.put(self.namespace_target, "current_task", {"description": initial_task})
        
        # 初始化库存
        self.store.put(self.namespace_inventory, "items", {"list": []})
        
        # 初始化 OpenAI 客户端
        self.openai_client = OpenAI(
            base_url="http://127.0.0.1:8080",
            api_key="EMPTY"
        )
        self.model_name = "Qwen3.5-9B"
        self.conversation_history = []
        
    def update_task(self, new_task: str):
        """更新当前任务"""
        self.store.put(self.namespace_target, "current_task", {"description": new_task})
        print(f"[系统] 任务已更新: {new_task}")
    
    def clear_task(self):
        """任务完成后清除"""
        self.store.delete(self.namespace_target, "current_task")
        print("[系统] 任务已清除")
    
    def add_to_inventory(self, item: str):
        """添加物体到库存"""
        items = self.get_inventory()
        items.append(item)
        self.store.put(self.namespace_inventory, "items", {"list": items})
        print(f"[系统] 已抓取: {item}")
    
    def remove_from_inventory(self, item: str):
        """从库存移除物体"""
        items = self.get_inventory()
        if item in items:
            items.remove(item)
            self.store.put(self.namespace_inventory, "items", {"list": items})
            print(f"[系统] 已释放: {item}")
    
    def get_inventory(self) -> list:
        """获取当前库存"""
        item = self.store.get(self.namespace_inventory, "items")
        return item.value["list"] if item else []
    
    def get_core_prompt(self) -> str:
        item = self.store.get(self.namespace_core, "prompt")
        return item.value["content"] if item else ""
    
    def get_current_task(self) -> Optional[str]:
        item = self.store.get(self.namespace_target, "current_task")
        return item.value["description"] if item else None
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取 JSON"""
        if not text:
            return None
            
        # 方法1: 直接解析
        try:
            return json.loads(text)
        except:
            pass
        
        # 方法2: 提取 {...} 中的内容
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # 方法3: 尝试修复常见的 JSON 错误
        try:
            # 替换单引号为双引号
            text_fixed = re.sub(r"'([^']*)'", r'"\1"', text)
            json_match = re.search(r'\{.*\}', text_fixed, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # 方法4: 手动提取键值对
        try:
            result = {}
            # 查找 action
            action_match = re.search(r'action["\']?\s*:\s*["\']?(\w+)["\']?', text, re.IGNORECASE)
            if action_match:
                result["action"] = action_match.group(1).upper()
            
            # 查找 speed
            speed_match = re.search(r'speed["\']?\s*:\s*([0-9.]+)', text, re.IGNORECASE)
            if speed_match:
                result["speed"] = float(speed_match.group(1))
            
            # 查找 duration
            duration_match = re.search(r'duration["\']?\s*:\s*([0-9.]+)', text, re.IGNORECASE)
            if duration_match:
                result["duration"] = float(duration_match.group(1))
            
            # 查找 target_object
            obj_match = re.search(r'target_object["\']?\s*:\s*["\']?([^"\',}]*)["\']?', text, re.IGNORECASE)
            if obj_match:
                result["target_object"] = obj_match.group(1).strip()
            
            # 查找 reason
            reason_match = re.search(r'reason["\']?\s*:\s*["\']?([^"\',}]*)["\']?', text, re.IGNORECASE)
            if reason_match:
                result["reason"] = reason_match.group(1).strip()
            
            return result if result else None
        except:
            return None
    
    def decide_command(self, sensor_data: dict, arm_angle: float = 0.0, inventory: list = None) -> Optional[RobotCommand]:
        """根据传感器数据决策下一步指令"""
        core = self.get_core_prompt()
        task = self.get_current_task()
        inventory = inventory or self.get_inventory()
        
        if not core:
            raise ValueError("核心提示词未设置")
        
        # 构建系统消息
        inventory_str = ", ".join(inventory) if inventory else "无"
        system_content = f"""{core}

当前任务：{task if task else '无任务，保持空闲'}
当前库存：{inventory_str}
机械臂角度：{arm_angle:.1f}度

【重要】请基于传感器数据输出一个JSON格式的指令，格式如下：
{{
    "action": "动作名称",
    "speed": 0.5,
    "duration": 1.0,
    "target_object": "目标物体名称（抓取相关动作需要）",
    "reason": "决策原因"
}}

可用动作：
- 移动类：MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, STOP
- 避障类：AVOID_OBSTACLE, RESUME_TASK
- 抓取类：SCAN_FOR_OBJECTS（扫描周围物体）, APPROACH_OBJECT（靠近物体）, 
         GRAB_OBJECT（抓取物体）, RELEASE_OBJECT（释放物体）
- 机械臂：MOVE_ARM_UP, MOVE_ARM_DOWN（调整机械臂高度）

规则：
1. 避障优先级最高：当任何方向距离 < 0.5米时，必须输出AVOID_OBSTACLE
2. 避障完成后（所有方向距离 > 0.8米），输出RESUME_TASK
3. 抓取流程：
   - 首先SCAN_FOR_OBJECTS识别周围物体
   - 发现目标物体后，APPROACH_OBJECT靠近（距离<1.0米）
   - 调整机械臂MOVE_ARM_UP/DOWN到合适高度
   - GRAB_OBJECT抓取物体（成功后会加入库存）
   - 需要放置时RELEASE_OBJECT释放
4. 正常情况向目标移动
5. 只输出JSON，不要添加任何其他文字
"""
        
        # 构建用户消息（传感器数据 + 物体信息）
        objects_info = ""
        if "objects" in sensor_data and sensor_data["objects"]:
            objects_info = "\n检测到的物体：\n"
            for obj in sensor_data["objects"]:
                objects_info += f"- {obj['name']}: 距离{obj['distance']:.2f}米, 位置({obj['x']:.1f}, {obj['y']:.1f})\n"
        
        sensor_str = f"""传感器数据：
- 前方: {sensor_data.get('front', 0):.2f}米
- 左侧: {sensor_data.get('left', 0):.2f}米
- 右侧: {sensor_data.get('right', 0):.2f}米
- 到目标距离: {sensor_data.get('target_distance', 0):.2f}米
{objects_info}
机械臂当前角度: {arm_angle:.1f}度"""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": sensor_str}
        ]
        
        try:
            print(f"\n[发送请求]")
            print(f"系统消息长度: {len(system_content)} 字符")
            print(f"传感器数据: {sensor_str[:200]}...")
            
            # 调用模型
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            print(f"[模型原始响应] {content}")
            
            # 使用增强的 JSON 提取
            data = self.extract_json(content)
            
            if not data:
                print("[警告] 无法解析JSON，尝试使用默认值")
                # 根据传感器数据做简单决策
                if sensor_data.get('front', 10) < 0.5:
                    data = {"action": "AVOID_OBSTACLE", "speed": 0.3, "duration": 1.0, "reason": "前方有障碍，自动避障"}
                elif sensor_data.get('left', 10) < 0.5:
                    data = {"action": "TURN_RIGHT", "speed": 0.3, "duration": 1.0, "reason": "左侧有障碍，向右转"}
                elif sensor_data.get('right', 10) < 0.5:
                    data = {"action": "TURN_LEFT", "speed": 0.3, "duration": 1.0, "reason": "右侧有障碍，向左转"}
                elif sensor_data.get('objects') and len(sensor_data['objects']) > 0:
                    # 如果检测到物体，尝试抓取
                    closest_obj = min(sensor_data['objects'], key=lambda x: x['distance'])
                    if closest_obj['distance'] < 2.0:
                        data = {"action": "APPROACH_OBJECT", "target_object": closest_obj['name'], 
                               "speed": 0.3, "duration": 1.0, "reason": f"发现{closest_obj['name']}，准备靠近"}
                    else:
                        data = {"action": "MOVE_FORWARD", "speed": 0.5, "duration": 1.0, "reason": "无障碍，向目标前进"}
                else:
                    data = {"action": "MOVE_FORWARD", "speed": 0.5, "duration": 1.0, "reason": "无障碍，向目标前进"}
            
            # 确保必要字段存在
            if "action" not in data:
                data["action"] = "STOP"
            if "speed" not in data:
                data["speed"] = 0.5
            if "duration" not in data:
                data["duration"] = 1.0
            
            # 确保action是大写的
            data["action"] = data["action"].upper()
            
            # 创建指令对象
            command = RobotCommand(**data)
            print(f"[解析结果] {command}")
            
            # 保存到历史
            self.conversation_history.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
            
            return command
            
        except Exception as e:
            print(f"[错误] 模型调用或解析失败：{e}")
            # 返回基于传感器数据的默认指令
            if sensor_data.get('front', 10) < 0.5:
                return RobotCommand(action="AVOID_OBSTACLE", speed=0.3, duration=1.0, reason=f"错误后默认避障: {str(e)[:50]}")
            else:
                return RobotCommand(action="MOVE_FORWARD", speed=0.3, duration=1.0, reason=f"错误后默认前进: {str(e)[:50]}")

# 为了兼容 LangGraph，定义状态字典
class GraphState(TypedDict):
    sensor_data: dict
    task_status: str
    last_command: Optional[dict]
    error: Optional[str]
    inventory: list
    arm_angle: float

def sensor_input_node(state: GraphState) -> GraphState:
    """模拟传感器输入"""
    return state

def decision_node(state: GraphState, brain: MemoryandRead) -> GraphState:
    """调用大脑决策"""
    try:
        cmd = brain.decide_command(
            state["sensor_data"], 
            arm_angle=state.get("arm_angle", 0.0),
            inventory=state.get("inventory", [])
        )
        if cmd:
            print(f"发出指令：{cmd.action}, 速度={cmd.speed}, 目标={cmd.target_object}, 原因={cmd.reason}")
            state["last_command"] = cmd.model_dump()
            
            # 更新任务状态
            if cmd.action == "RESUME_TASK":
                state["task_status"] = "RUNNING"
            elif cmd.action == "AVOID_OBSTACLE":
                state["task_status"] = "AVOIDING"
            elif cmd.action == "GRAB_OBJECT":
                state["task_status"] = "GRABBING"
            elif cmd.action == "RELEASE_OBJECT":
                state["task_status"] = "RELEASING"
            elif cmd.action == "STOP" and state["task_status"] == "RUNNING":
                state["task_status"] = "COMPLETED"
                brain.clear_task()
        state["error"] = None
    except Exception as e:
        state["error"] = str(e)
    return state

def should_continue(state: GraphState) -> str:
    """决定是否结束循环"""
    if state.get("error"):
        return "error"
    return "continue"

# 构建图
builder = StateGraph(GraphState)
builder.add_node("sensor_input", sensor_input_node)
builder.add_node("decide", decision_node)
builder.set_entry_point("sensor_input")
builder.add_edge("sensor_input", "decide")
builder.add_conditional_edges("decide", should_continue, {
    "continue": "sensor_input",
    "error": END
})
graph = builder.compile(checkpointer=MemorySaver())

# ==================== 模拟机器人 ====================

class SimulatedRobot:
    """模拟机器人，包含传感器、运动和机械臂"""
    
    def __init__(self, start_pos=(0, 0), target=(10, 10)):
        self.position = list(start_pos)
        self.target = target
        self.direction = 0
        self.speed = 0.5
        self.arm_angle = 0.0  # 机械臂角度（-45到45度）
        self.inventory = []  # 抓取的物体
        
        # 障碍物
        self.obstacles = [
            {"pos": (3, 0), "radius": 1.0},
            {"pos": (5, 5), "radius": 1.5},
            {"pos": (8, 2), "radius": 0.8},
        ]
        
        # 可抓取的物体
        self.objects = [
            {"name": "cube", "pos": (4, 2), "radius": 0.3, "grabbed": False},
            {"name": "sphere", "pos": (6, 3), "radius": 0.4, "grabbed": False},
            {"name": "cylinder", "pos": (7, 7), "radius": 0.3, "grabbed": False},
            {"name": "box", "pos": (2, 8), "radius": 0.5, "grabbed": False},
        ]
        
        self.step_count = 0
        self.command_history = []
        self.nearby_object = None  # 当前靠近的物体
        
    def read_sensors(self) -> Dict[str, Any]:
        """模拟读取传感器数据，包括物体检测"""
        # 基础距离传感器
        front_dist = 10.0
        left_dist = 10.0
        right_dist = 10.0
        
        # 检查到目标的距离
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        target_dist = (dx**2 + dy**2)**0.5
        
        # 模拟障碍物检测
        for obs in self.obstacles:
            obs_dx = obs["pos"][0] - self.position[0]
            obs_dy = obs["pos"][1] - self.position[1]
            obs_dist = (obs_dx**2 + obs_dy**2)**0.5 - obs["radius"]
            
            # 简化：根据位置分配传感器
            if abs(obs_dx) < 2 and abs(obs_dy) < 2:
                angle_to_obs = math.degrees(math.atan2(obs_dy, obs_dx)) - self.direction
                angle_to_obs = (angle_to_obs + 360) % 360
                
                if angle_to_obs < 45 or angle_to_obs > 315:
                    front_dist = min(front_dist, obs_dist)
                elif angle_to_obs < 135:
                    left_dist = min(left_dist, obs_dist)
                else:
                    right_dist = min(right_dist, obs_dist)
        
        # 检测可抓取物体
        detected_objects = []
        for obj in self.objects:
            if obj["grabbed"]:
                continue
                
            obj_dx = obj["pos"][0] - self.position[0]
            obj_dy = obj["pos"][1] - self.position[1]
            obj_dist = (obj_dx**2 + obj_dy**2)**0.5
            
            # 如果在检测范围内（5米）
            if obj_dist < 5.0:
                detected_objects.append({
                    "name": obj["name"],
                    "distance": obj_dist,
                    "x": obj_dx,
                    "y": obj_dy,
                    "angle": math.degrees(math.atan2(obj_dy, obj_dx)) - self.direction
                })
        
        # 找出最近的物体
        if detected_objects:
            self.nearby_object = min(detected_objects, key=lambda x: x["distance"])
        else:
            self.nearby_object = None
        
        return {
            "front": max(0, front_dist),
            "left": max(0, left_dist),
            "right": max(0, right_dist),
            "target_distance": target_dist,
            "objects": detected_objects
        }
    
    def execute_command(self, command: RobotCommand) -> tuple:
        """
        执行机器人指令
        返回: (是否完成任务, 是否抓取成功, 消息)
        """
        self.command_history.append(command)
        
        print(f"\n[执行] 指令: {command.action}")
        print(f"       原因: {command.reason}")
        if command.target_object:
            print(f"       目标物体: {command.target_object}")
        
        # 处理移动指令
        if command.action == "MOVE_FORWARD":
            rad = math.radians(self.direction)
            self.position[0] += command.speed * command.duration * math.cos(rad)
            self.position[1] += command.speed * command.duration * math.sin(rad)
            
        elif command.action == "MOVE_BACKWARD":
            rad = math.radians(self.direction + 180)
            self.position[0] += command.speed * command.duration * math.cos(rad)
            self.position[1] += command.speed * command.duration * math.sin(rad)
            
        elif command.action == "TURN_LEFT":
            self.direction = (self.direction + 90) % 360
            
        elif command.action == "TURN_RIGHT":
            self.direction = (self.direction - 90) % 360
            
        elif command.action == "AVOID_OBSTACLE":
            # 简单避障：随机转向
            self.direction = (self.direction + random.choice([-90, 90])) % 360
            # 向前移动一小步
            rad = math.radians(self.direction)
            self.position[0] += 0.3 * math.cos(rad)
            self.position[1] += 0.3 * math.sin(rad)
            
        elif command.action == "MOVE_ARM_UP":
            self.arm_angle = min(45, self.arm_angle + 15)
            print(f"   机械臂上升至 {self.arm_angle}度")
            
        elif command.action == "MOVE_ARM_DOWN":
            self.arm_angle = max(-45, self.arm_angle - 15)
            print(f"   机械臂下降至 {self.arm_angle}度")
            
        elif command.action == "SCAN_FOR_OBJECTS":
            # 扫描动作已经通过传感器完成
            if self.nearby_object:
                print(f"   扫描到物体: {self.nearby_object['name']} 距离 {self.nearby_object['distance']:.2f}米")
            else:
                print("   未扫描到物体")
                
        elif command.action == "APPROACH_OBJECT":
            if self.nearby_object:
                # 朝向物体移动
                obj_angle = self.nearby_object['angle']
                if abs(obj_angle) > 10:
                    # 先转向物体
                    if obj_angle > 0:
                        self.direction = (self.direction + 15) % 360
                    else:
                        self.direction = (self.direction - 15) % 360
                else:
                    # 向物体移动
                    rad = math.radians(self.direction)
                    self.position[0] += 0.2 * math.cos(rad)
                    self.position[1] += 0.2 * math.sin(rad)
                print(f"   靠近物体: {self.nearby_object['name']}, 距离 {self.nearby_object['distance']:.2f}米")
            else:
                print("   没有可靠近的物体")
                
        elif command.action == "GRAB_OBJECT":
            target = command.target_object
            grabbed = False
            
            for obj in self.objects:
                if obj["name"] == target and not obj["grabbed"]:
                    # 检查是否在抓取范围内
                    obj_dx = obj["pos"][0] - self.position[0]
                    obj_dy = obj["pos"][1] - self.position[1]
                    obj_dist = (obj_dx**2 + obj_dy**2)**0.5
                    
                    if obj_dist < 1.0 and abs(self.arm_angle) < 30:
                        obj["grabbed"] = True
                        self.inventory.append(target)
                        grabbed = True
                        print(f"   ✅ 成功抓取 {target}！")
                        break
            
            if not grabbed:
                print(f"   ❌ 抓取失败: {target} 不在范围内或机械臂角度不合适")
                
        elif command.action == "RELEASE_OBJECT":
            target = command.target_object
            if target in self.inventory:
                self.inventory.remove(target)
                # 将物体放在当前位置附近
                for obj in self.objects:
                    if obj["name"] == target:
                        obj["grabbed"] = False
                        obj["pos"] = (self.position[0] + 0.5, self.position[1])
                        break
                print(f"   释放物体 {target}")
            else:
                print(f"   库存中没有 {target}")
        
        self.step_count += 1
        
        # 检查是否到达目标
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        dist_to_target = (dx**2 + dy**2)**0.5
        
        # 显示当前状态
        print(f"   位置: ({self.position[0]:.2f}, {self.position[1]:.2f})")
        print(f"   方向: {self.direction}°")
        print(f"   机械臂: {self.arm_angle:.1f}°")
        print(f"   库存: {self.inventory}")
        
        if dist_to_target < 0.5:
            print(f"\n🎉 成功到达目标位置！")
            return True, False, "到达目标"
        
        return False, False, "继续"

# ==================== 测试函数 ====================

def test_robot_with_grasping():
    """测试机器人抓取功能"""
    
    core_prompt = """
你是一个自主移动机器人的决策核心，配备机械臂可以抓取物体。
根据传感器数据输出JSON格式的运动指令。

可用动作：
- 移动类：MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, STOP
- 避障类：AVOID_OBSTACLE, RESUME_TASK
- 抓取类：SCAN_FOR_OBJECTS, APPROACH_OBJECT, GRAB_OBJECT, RELEASE_OBJECT
- 机械臂：MOVE_ARM_UP, MOVE_ARM_DOWN

规则：
1. 避障优先级最高：距离 < 0.5米时立即避障
2. 抓取流程：
   - 首先SCAN_FOR_OBJECTS识别周围物体
   - 发现目标后APPROACH_OBJECT靠近（距离<1.0米）
   - 调整机械臂MOVE_ARM_UP/DOWN到合适高度
   - GRAB_OBJECT抓取物体
   - 需要放置时RELEASE_OBJECT
3. 任务可能是："移动到目标点"或"抓取特定物体"
4. 只输出JSON，不要其他文字
"""
    
    # 初始化
    brain = MemoryandRead(core_prompt, initial_task="抓取cube并移动到目标点(10,10)")
    robot = SimulatedRobot(start_pos=(0, 0), target=(10, 10))
    
    print("=" * 60)
    print("机器人抓取测试开始")
    print(f"起始位置: {robot.position}")
    print(f"目标位置: {robot.target}")
    print(f"可抓取物体: {[obj['name'] for obj in robot.objects]}")
    print("=" * 60)
    
    max_steps = 50
    task_completed = False
    
    for step in range(max_steps):
        print(f"\n--- 步骤 {step + 1} ---")
        
        # 读取传感器
        sensor_data = robot.read_sensors()
        print(f"传感器: 前={sensor_data['front']:.2f}m, 左={sensor_data['left']:.2f}m, 右={sensor_data['right']:.2f}m")
        if sensor_data['objects']:
            print(f"检测到物体: {[obj['name'] for obj in sensor_data['objects']]}")
        
        # 大脑决策
        command = brain.decide_command(
            sensor_data,
            arm_angle=robot.arm_angle,
            inventory=robot.inventory
        )
        
        if not command:
            print("[错误] 无法获得有效指令")
            break
        
        # 执行指令
        completed, grabbed, msg = robot.execute_command(command)
        
        # 如果抓取成功，更新库存
        if command.action == "GRAB_OBJECT" and command.target_object in robot.inventory:
            brain.add_to_inventory(command.target_object)
        
        if completed:
            task_completed = True
            brain.clear_task()
            break
        
        time.sleep(0.5)
    
    # 测试结果总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if task_completed:
        print("✅ 任务成功完成！")
    else:
        print("❌ 任务未完成（达到最大步数）")
    
    print(f"最终位置: ({robot.position[0]:.2f}, {robot.position[1]:.2f})")
    print(f"最终库存: {robot.inventory}")
    print(f"执行步数: {robot.step_count}")
    print(f"机械臂角度: {robot.arm_angle:.1f}°")

def test_api_connection():
    """测试 API 连接"""
    print("=" * 60)
    print("测试 API 连接")
    print("=" * 60)
    
    client = OpenAI(
        base_url="http://127.0.0.1:8080",
        api_key="EMPTY"
    )
    
    try:
        response = client.chat.completions.create(
            model="Qwen3.5-9B",
            messages=[
                {"role": "system", "content": "你是一个助手。输出JSON。"},
                {"role": "user", "content": "输出：{\"test\": \"hello\"}"}
            ],
            max_tokens=50
        )
        print("✅ API连接成功")
        print(f"响应: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return False

if __name__ == "__main__":
    import math
    
    # 先测试 API 连接
    if test_api_connection():
        # 运行主要测试
        test_robot_with_grasping()
    else:
        print("\n请检查 API 服务是否正常运行在 http://127.0.0.1:8080")