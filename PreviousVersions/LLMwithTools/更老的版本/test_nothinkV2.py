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

# 模型输出结构化指令
class RobotCommand(BaseModel):
    action: Literal["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP", "AVOID_OBSTACLE", "RESUME_TASK"]
    speed: Optional[float] = 0.5
    duration: Optional[float] = 1.0
    reason: Optional[str] = None

# 机器人状态（LangGraph 状态）
class RobotState(BaseModel):
    sensor_data: dict = {}
    task_status: str = "IDLE"
    last_command: Optional[RobotCommand] = None
    error: Optional[str] = None

class MemoryandRead:
    def __init__(self, core_prompt: str, initial_task: str = ""):
        self.store = InMemoryStore()
        self.namespace_core = ("robot", "core")
        self.namespace_target = ("robot", "target")
        
        # 存储核心提示词（永不更改）
        self.store.put(self.namespace_core, "prompt", {"content": core_prompt})
        
        # 存储初始任务
        if initial_task:
            self.store.put(self.namespace_target, "current_task", {"description": initial_task})
        
        # 初始化 OpenAI 客户端
        self.openai_client = OpenAI(
            base_url="http://127.0.0.1:8080",
            api_key="EMPTY"
        )
        self.model_name = "Qwen3.5-9B"
        self.conversation_history = []
        
    def update_task(self, new_task: str):
        """更新当前任务（覆盖）"""
        self.store.put(self.namespace_target, "current_task", {"description": new_task})
        print(f"[系统] 任务已更新: {new_task}")
    
    def clear_task(self):
        """任务完成后清除"""
        self.store.delete(self.namespace_target, "current_task")
        print("[系统] 任务已清除")
    
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
            
            # 查找 reason
            reason_match = re.search(r'reason["\']?\s*:\s*["\']?([^"\',}]*)["\']?', text, re.IGNORECASE)
            if reason_match:
                result["reason"] = reason_match.group(1).strip()
            
            return result if result else None
        except:
            return None
    
    def decide_command(self, sensor_data: dict, recent_history: list = None) -> Optional[RobotCommand]:
        """根据传感器数据决策下一步指令"""
        core = self.get_core_prompt()
        task = self.get_current_task()
        
        if not core:
            raise ValueError("核心提示词未设置")
        
        # 【关键修复】将所有内容整合到一个系统消息中
        system_content = f"""{core}

当前任务：{task if task else '无任务，保持空闲'}

【重要】请基于传感器数据输出一个JSON格式的指令，格式如下：
{{
    "action": "动作名称",
    "speed": 0.5,
    "duration": 1.0,
    "reason": "决策原因"
}}

可用动作：MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, STOP, AVOID_OBSTACLE, RESUME_TASK

规则：
1. 当任何方向距离 < 0.5米时，必须输出AVOID_OBSTACLE
2. 避障完成后（所有方向距离 > 0.8米），输出RESUME_TASK
3. 正常情况下向目标移动
4. 只输出JSON，不要添加任何其他文字
"""
        
        # 构建用户消息（传感器数据）
        sensor_str = f"""传感器数据：
- 前方: {sensor_data.get('front', 0):.2f}米
- 左侧: {sensor_data.get('left', 0):.2f}米
- 右侧: {sensor_data.get('right', 0):.2f}米
- 到目标距离: {sensor_data.get('target_distance', 0):.2f}米"""

        # 严格按照要求：只有一个系统消息和一个用户消息
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": sensor_str}
        ]
        
        try:
            print(f"\n[发送请求]")
            print(f"系统消息长度: {len(system_content)} 字符")
            print(f"传感器数据: {sensor_str}")
            
            # 调用模型
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,  # 降低温度使输出更稳定
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            print(f"[模型原始响应] {content}")
            
            # 使用增强的 JSON 提取
            data = self.extract_json(content)
            
            if not data:
                print("[警告] 无法解析JSON，尝试使用默认值")
                # 根据传感器数据做出简单决策
                if sensor_data.get('front', 10) < 0.5:
                    data = {"action": "AVOID_OBSTACLE", "speed": 0.3, "duration": 1.0, "reason": "前方有障碍，自动避障"}
                elif sensor_data.get('left', 10) < 0.5:
                    data = {"action": "TURN_RIGHT", "speed": 0.3, "duration": 1.0, "reason": "左侧有障碍，向右转"}
                elif sensor_data.get('right', 10) < 0.5:
                    data = {"action": "TURN_LEFT", "speed": 0.3, "duration": 1.0, "reason": "右侧有障碍，向左转"}
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

def sensor_input_node(state: GraphState) -> GraphState:
    """模拟传感器输入"""
    return state

def decision_node(state: GraphState, brain: MemoryandRead) -> GraphState:
    """调用大脑决策"""
    try:
        cmd = brain.decide_command(state["sensor_data"])
        if cmd:
            print(f"发出指令：{cmd.action}, 速度={cmd.speed}, 原因={cmd.reason}")
            state["last_command"] = cmd.model_dump()
            
            # 更新任务状态
            if cmd.action == "RESUME_TASK":
                state["task_status"] = "RUNNING"
            elif cmd.action == "AVOID_OBSTACLE":
                state["task_status"] = "AVOIDING"
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
    """模拟机器人，包含传感器和运动"""
    
    def __init__(self, start_pos=(0, 0), target=(10, 10)):
        self.position = list(start_pos)
        self.target = target
        self.direction = 0
        self.speed = 0.5
        self.obstacles = [
            {"pos": (3, 0), "radius": 1.0},
            {"pos": (5, 5), "radius": 1.5},
            {"pos": (8, 2), "radius": 0.8},
        ]
        self.step_count = 0
        self.command_history = []
        
    def read_sensors(self) -> Dict[str, float]:
        """模拟读取传感器数据"""
        # 简化的传感器模拟
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
                if abs(obs_dy) < abs(obs_dx):
                    front_dist = min(front_dist, obs_dist)
                elif obs_dy > 0:
                    left_dist = min(left_dist, obs_dist)
                else:
                    right_dist = min(right_dist, obs_dist)
        
        return {
            "front": max(0, front_dist),
            "left": max(0, left_dist),
            "right": max(0, right_dist),
            "target_distance": target_dist
        }
    
    def execute_command(self, command: RobotCommand) -> bool:
        """执行机器人指令"""
        self.command_history.append(command)
        
        print(f"\n[执行] 指令: {command.action}")
        print(f"       原因: {command.reason}")
        
        # 根据指令移动
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
        
        self.step_count += 1
        
        # 检查是否到达目标
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        dist_to_target = (dx**2 + dy**2)**0.5
        
        if dist_to_target < 0.5:
            print(f"\n🎉 成功到达目标位置！")
            return True
        
        return False

# ==================== 测试函数 ====================

def test_robot_brain():
    """测试机器人大脑功能"""
    
    core_prompt = """
你是一个自主移动机器人的决策核心。
根据传感器数据输出JSON格式的运动指令。

可用动作：
- MOVE_FORWARD: 向前移动
- MOVE_BACKWARD: 向后移动
- TURN_LEFT: 左转
- TURN_RIGHT: 右转
- STOP: 停止
- AVOID_OBSTACLE: 避障
- RESUME_TASK: 恢复任务

规则：
- 距离 < 0.5米时避障
- 避障完成后恢复任务
- 只输出JSON，不要其他文字
"""
    
    # 初始化
    brain = MemoryandRead(core_prompt, initial_task="移动到位置(10, 10)")
    robot = SimulatedRobot(start_pos=(0, 0), target=(10, 10))
    
    print("=" * 60)
    print("机器人大脑测试开始")
    print(f"起始位置: {robot.position}")
    print(f"目标位置: {robot.target}")
    print("=" * 60)
    
    max_steps = 30
    task_completed = False
    
    for step in range(max_steps):
        print(f"\n--- 步骤 {step + 1} ---")
        print(f"当前位置: ({robot.position[0]:.2f}, {robot.position[1]:.2f})")
        print(f"当前方向: {robot.direction}°")
        
        # 读取传感器
        sensor_data = robot.read_sensors()
        print(f"传感器: 前={sensor_data['front']:.2f}m, 左={sensor_data['left']:.2f}m, 右={sensor_data['right']:.2f}m")
        
        # 大脑决策
        command = brain.decide_command(sensor_data)
        
        if not command:
            print("[错误] 无法获得有效指令")
            break
        
        # 执行指令
        completed = robot.execute_command(command)
        
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
    print(f"执行步数: {robot.step_count}")

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
        # 最简单的消息格式
        response = client.chat.completions.create(
            model="Qwen3.5-9B",
            messages=[
                {"role": "system", "content": "你是一个助手。输出JSON。"},
                {"role": "user", "content": "输出：{\"test\": \"hello\"}"}
            ],
            max_tokens=50
        )
        print("✅ API 连接成功")
        print(f"响应: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ API 连接失败: {e}")
        return False

if __name__ == "__main__":
    import math
    
    # 先测试 API 连接
    if test_api_connection():
        # 运行主要测试
        test_robot_brain()
    else:
        print("\n请检查 API 服务是否正常运行在 http://127.0.0.1:8080")