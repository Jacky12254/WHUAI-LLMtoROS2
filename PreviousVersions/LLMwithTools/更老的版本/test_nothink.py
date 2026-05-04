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
from typing import Dict, Any, Optional

# 模型输出结构化指令
class RobotCommand(BaseModel):
    action: Literal["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP", "AVOID_OBSTACLE", "RESUME_TASK"]
    speed: Optional[float] = 0.5          # 移动速度
    duration: Optional[float] = 1.0        # 动作持续时间（秒）
    reason: Optional[str] = None            # 决策原因，便于调试

# 机器人状态（LangGraph 状态）
class RobotState(BaseModel):
    sensor_data: dict = {}                  # 传感器读数，例如 {"front_distance": 0.3, "left_distance": 0.8}
    task_status: str = "IDLE"                # IDLE, RUNNING, AVOIDING, COMPLETED
    last_command: Optional[RobotCommand] = None
    error: Optional[str] = None

# class MemoryandRead:
#     def __init__(self):
#         self.store = InMemoryStore()
#         self.namespace_core = ("core", "scratchpad")
#         self.namespace_target = ("target", "scratchpad")
        
#         self.openai_client = OpenAI(
#             base_url="http://127.0.0.1:8001/v1",
#             api_key="sk-no-key-required",
#             )
        
        
#     def init_model(self):
#         self.completion = self.openai_client.chat.completions.create(
#             model="unsloth/Qwen3.5-397B-A17B",
#             messages=[{"role": "system", "content": "Create a Snake game."}],
#             )
#         self.store.put(self.namespace_core, "notes", {"scratchpad": self.completion.choices[0].message.content})

#     def input_prompt(self):
#         self.completion = self.openai_client.chat.completions.create(
#             model="unsloth/Qwen3.5-397B-A17B",
#             messages=[{"role": "user", "content": "Create a Snake game."}],
#             )
        
#     def output_answer(self):
#         return self.completion.choices[0].message.content

#     def tool_node(self, state):
#         tool_call = state["messages"][-1].tool_calls[0]

#         if tool_call["name"] == "WriteToScratchpad":
#             self.store.put(self.namespace_target, "notes", {"scratchpad": tool_call["args"]["data"]})

#         if tool_call["name"] == "ReadFromScratchpad":
#             saved = self.store.get(self.namespace_target, "notes")
#             return saved["scratchpad"] if saved else "No notes found"
        
#     def tranform_to_order(self):
#         answer = json.loads(self.output_answer())
#         location = answer["location"]
#         arm_location = answer["arm_location"]
#         return location, arm_location

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
        
    def update_task(self, new_task: str):
        """更新当前任务（覆盖）"""
        self.store.put(self.namespace_target, "current_task", {"description": new_task})
    
    def clear_task(self):
        """任务完成后清除"""
        self.store.delete(self.namespace_target, "current_task")
    
    def get_core_prompt(self) -> str:
        item = self.store.get(self.namespace_core, "prompt")
        return item.value["content"] if item else ""
    
    def get_current_task(self) -> Optional[str]:
        item = self.store.get(self.namespace_target, "current_task")
        return item.value["description"] if item else None
    
    def decide_command(self, sensor_data: dict, recent_history: list = None) -> Optional[RobotCommand]:
        """根据传感器数据决策下一步指令"""
        core = self.get_core_prompt()
        task = self.get_current_task()
        
        if not core:
            raise ValueError("核心提示词未设置")
        
        # 构建消息
        messages = [
            {"role": "system", "content": core}
        ]
        
        # 添加任务信息
        if task:
            messages.append({"role": "system", "content": f"当前任务：{task}"})
        else:
            messages.append({"role": "system", "content": "当前无任务，请保持空闲状态"})
        
        # 添加上下文历史（最近3条）
        if recent_history:
            for msg in recent_history[-3:]:
                messages.append(msg)
        
        # 添加当前传感器数据
        sensor_str = f"传感器数据：前方={sensor_data.get('front', 0):.2f}m, 左侧={sensor_data.get('left', 0):.2f}m, 右侧={sensor_data.get('right', 0):.2f}m"
        messages.append({"role": "user", "content": sensor_str})
        
        # 要求输出JSON格式
        messages.append({"role": "system", "content": "请输出JSON格式指令：{\"action\": \"...\", \"speed\": 0.5, \"duration\": 1.0, \"reason\": \"...\"}"})
        
        try:
            # 调用模型
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            content = response.choices[0].message.content
            print(f"[模型响应] {content}")
            
            # 解析JSON
            # 尝试提取JSON部分（防止模型输出多余文本）
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            data = json.loads(content)
            command = RobotCommand(**data)
            
            # 保存到历史
            self.conversation_history.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
            
            return command
        except Exception as e:
            print(f"[错误] 模型调用或解析失败：{e}")
            # 返回默认停止指令
            return RobotCommand(action="STOP", reason=f"Error: {str(e)}")

# 为了兼容 LangGraph，定义状态字典（也可直接用 Pydantic）
class GraphState(TypedDict):
    sensor_data: dict
    task_status: str
    last_command: Optional[dict]
    error: Optional[str]

def sensor_input_node(state: GraphState) -> GraphState:
    """模拟传感器输入（实际中可从硬件读取）"""
    # 这里简化，实际可从外部传入或定时更新
    # 假设传感器数据已经更新到 state['sensor_data']
    return state

def decision_node(state: GraphState, brain: MemoryandRead) -> GraphState:
    """调用大脑决策"""
    try:
        cmd = brain.decide_command(state["sensor_data"])
        if cmd:
            # 发送指令给运动部件（此处仅记录）
            print(f"发出指令：{cmd.action}, 速度={cmd.speed}, 原因={cmd.reason}")
            state["last_command"] = cmd.model_dump()
            
            # 更新任务状态（可根据指令判断）
            if cmd.action == "RESUME_TASK":
                state["task_status"] = "RUNNING"
            elif cmd.action == "AVOID_OBSTACLE":
                state["task_status"] = "AVOIDING"
            elif cmd.action == "STOP" and state["task_status"] == "RUNNING":
                # 可能是任务完成
                state["task_status"] = "COMPLETED"
                brain.clear_task()  # 清除任务
        state["error"] = None
    except Exception as e:
        state["error"] = str(e)
    return state

def should_continue(state: GraphState) -> str:
    """决定是否结束循环（通常不会结束，除非发生致命错误）"""
    if state.get("error"):
        return "error"
    # 可根据需要决定结束条件，比如电池低电量等
    return "continue"

# 构建图
builder = StateGraph(GraphState)

# 添加节点
builder.add_node("sensor_input", sensor_input_node)
builder.add_node("decide", decision_node)

# 设置边
builder.set_entry_point("sensor_input")
builder.add_edge("sensor_input", "decide")
builder.add_conditional_edges("decide", should_continue, {
    "continue": "sensor_input",   # 循环继续
    "error": END
})

# 编译图（使用内存检查点以便追踪状态）
graph = builder.compile(checkpointer=MemorySaver())

# ==================== 3. 模拟环境和运动部件 ====================

class SimulatedRobot:
    """模拟机器人，包含传感器和运动"""
    
    def __init__(self, start_pos=(0, 0), target=(10, 10)):
        self.position = list(start_pos)
        self.target = target
        self.direction = 0  # 0: 右, 90: 上, 180: 左, 270: 下
        self.speed = 0.5
        self.obstacles = [
            {"pos": (3, 0), "radius": 1.0},   # 障碍物1
            {"pos": (5, 5), "radius": 1.5},   # 障碍物2
            {"pos": (8, 2), "radius": 0.8},   # 障碍物3
        ]
        self.step_count = 0
        self.command_history = []
        
    def read_sensors(self) -> Dict[str, float]:
        """模拟读取传感器数据"""
        # 计算各个方向的距离
        front_pos = self._get_front_position()
        left_pos = self._get_left_position()
        right_pos = self._get_right_position()
        
        front_dist = self._distance_to_nearest_obstacle(front_pos)
        left_dist = self._distance_to_nearest_obstacle(left_pos)
        right_dist = self._distance_to_nearest_obstacle(right_pos)
        
        # 添加一些随机噪声
        front_dist += random.uniform(-0.05, 0.05)
        left_dist += random.uniform(-0.05, 0.05)
        right_dist += random.uniform(-0.05, 0.05)
        
        # 限制最小值为0
        front_dist = max(0, front_dist)
        left_dist = max(0, left_dist)
        right_dist = max(0, right_dist)
        
        # 到目标的距离和方向
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        target_dist = (dx**2 + dy**2)**0.5
        target_angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360 if 'math' in globals() else 0
        
        return {
            "front": front_dist,
            "left": left_dist,
            "right": right_dist,
            "target_distance": target_dist,
            "target_angle": target_angle,
            "current_direction": self.direction
        }
    
    def _get_front_position(self):
        """获取前方1米处的坐标"""
        rad = math.radians(self.direction)
        return (self.position[0] + math.cos(rad), self.position[1] + math.sin(rad))
    
    def _get_left_position(self):
        """获取左侧1米处的坐标"""
        rad = math.radians(self.direction + 90)
        return (self.position[0] + math.cos(rad), self.position[1] + math.sin(rad))
    
    def _get_right_position(self):
        """获取右侧1米处的坐标"""
        rad = math.radians(self.direction - 90)
        return (self.position[0] + math.cos(rad), self.position[1] + math.sin(rad))
    
    def _distance_to_nearest_obstacle(self, point):
        """计算点到最近障碍物的距离"""
        min_dist = float('inf')
        for obs in self.obstacles:
            dx = point[0] - obs["pos"][0]
            dy = point[1] - obs["pos"][1]
            dist = (dx**2 + dy**2)**0.5 - obs["radius"]
            min_dist = min(min_dist, dist)
        return min_dist
    
    def execute_command(self, command: RobotCommand):
        """执行机器人指令"""
        self.command_history.append(command)
        
        print(f"\n[执行] 指令: {command.action}")
        print(f"       原因: {command.reason}")
        print(f"       速度: {command.speed}, 持续时间: {command.duration}")
        
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
            self.direction = (self.direction + 90 * command.duration) % 360
            
        elif command.action == "TURN_RIGHT":
            self.direction = (self.direction - 90 * command.duration) % 360
            
        elif command.action == "AVOID_OBSTACLE":
            # 避障模式：根据reason决定转向
            if "左转" in command.reason or "left" in command.reason.lower():
                self.direction = (self.direction + 90) % 360
            elif "右转" in command.reason or "right" in command.reason.lower():
                self.direction = (self.direction - 90) % 360
            # 向前移动一小步
            rad = math.radians(self.direction)
            self.position[0] += 0.3 * math.cos(rad)
            self.position[1] += 0.3 * math.sin(rad)
            
        elif command.action == "RESUME_TASK":
            print("[系统] 恢复执行原任务")
            
        elif command.action == "STOP":
            print("[系统] 停止")
        
        # 限制位置在合理范围内
        self.position[0] = max(-5, min(15, self.position[0]))
        self.position[1] = max(-5, min(15, self.position[1]))
        
        self.step_count += 1
        
        # 检查是否到达目标
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        dist_to_target = (dx**2 + dy**2)**0.5
        
        if dist_to_target < 0.5:
            print(f"\n🎉 成功到达目标位置！位置: ({self.position[0]:.2f}, {self.position[1]:.2f})")
            return True
        
        return False

# ==================== 4. 测试主函数 ====================

def test_robot_brain():
    """测试机器人大脑功能"""
    
    # 核心提示词
    core_prompt = """
你是一个自主移动机器人的决策核心。你的职责是：

1. 根据传感器数据（前方、左侧、右侧距离，单位米）决定运动指令。
2. 指令必须为JSON格式，包含action、speed、duration、reason字段。
3. 可能的action：
   - MOVE_FORWARD: 向前移动
   - MOVE_BACKWARD: 向后移动
   - TURN_LEFT: 左转
   - TURN_RIGHT: 右转
   - STOP: 停止
   - AVOID_OBSTACLE: 避障（在reason中说明具体动作）
   - RESUME_TASK: 避障完成后恢复任务

4. 避障规则：
   - 当任何方向距离 < 0.5米时，必须避障
   - 避障时输出AVOID_OBSTACLE，并通过reason说明转向方向
   - 避障完成后（所有方向距离 > 0.8米），输出RESUME_TASK

5. 任务优先级：
   - 始终以完成当前任务为最高目标
   - 当前任务信息会随每次输入提供

请始终保持输出格式正确，不要添加额外解释。
"""
    
    # 初始化大脑和模拟机器人
    brain = MemoryandRead(core_prompt, initial_task="移动到位置(10, 10)")
    robot = SimulatedRobot(start_pos=(0, 0), target=(10, 10))
    
    print("=" * 60)
    print("机器人大脑测试开始")
    print(f"起始位置: {robot.position}")
    print(f"目标位置: {robot.target}")
    print("=" * 60)
    
    max_steps = 50
    task_completed = False
    recent_history = []
    
    # 模拟运行循环
    for step in range(max_steps):
        print(f"\n--- 步骤 {step + 1} ---")
        print(f"当前位置: ({robot.position[0]:.2f}, {robot.position[1]:.2f})")
        print(f"当前方向: {robot.direction}°")
        
        # 1. 读取传感器
        sensor_data = robot.read_sensors()
        print(f"传感器: 前={sensor_data['front']:.2f}m, 左={sensor_data['left']:.2f}m, 右={sensor_data['right']:.2f}m")
        print(f"到目标距离: {sensor_data['target_distance']:.2f}m")
        
        # 2. 大脑决策
        command = brain.decide_command(sensor_data, recent_history)
        if not command:
            print("[错误] 无法获得有效指令")
            break
        
        # 3. 记录历史（用于上下文）
        recent_history.append({"role": "assistant", "content": json.dumps(command.dict())})
        if len(recent_history) > 5:
            recent_history.pop(0)
        
        # 4. 执行指令
        completed = robot.execute_command(command)
        
        # 5. 检查任务完成
        if completed:
            task_completed = True
            brain.clear_task()
            break
        
        # 模拟传感器更新间隔
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
    print(f"指令历史: {len(robot.command_history)}条")
    
    # 显示指令统计
    action_counts = {}
    for cmd in robot.command_history:
        action_counts[cmd.action] = action_counts.get(cmd.action, 0) + 1
    
    print("\n指令统计:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}次")

# ==================== 5. 测试任务更新功能 ====================

def test_task_updates():
    """测试任务更新功能"""
    print("\n" + "=" * 60)
    print("测试任务更新功能")
    print("=" * 60)
    
    core_prompt = "你是一个机器人决策核心。"
    brain = MemoryandRead(core_prompt)
    
    # 测试初始无任务
    task = brain.get_current_task()
    print(f"初始任务: {task}")
    
    # 设置任务1
    brain.update_task("移动到点A")
    task = brain.get_current_task()
    print(f"更新后任务: {task}")
    
    # 设置任务2（覆盖）
    brain.update_task("移动到点B")
    task = brain.get_current_task()
    print(f"再次更新: {task}")
    
    # 清除任务
    brain.clear_task()
    task = brain.get_current_task()
    print(f"清除后: {task}")

# ==================== 6. 测试异常处理 ====================

def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试错误处理")
    print("=" * 60)
    
    # 无核心提示词
    try:
        brain = MemoryandRead("")
        brain.decide_command({})
    except ValueError as e:
        print(f"✅ 正确捕获无核心提示词错误: {e}")
    
    # 无效传感器数据
    brain = MemoryandRead("你是一个机器人。")
    brain.update_task("测试任务")
    
    # 传入空数据
    cmd = brain.decide_command({})
    print(f"空传感器数据指令: {cmd}")
    
    # 传入异常数据
    cmd = brain.decide_command({"front": "invalid"})
    print(f"异常数据指令: {cmd}")

# ==================== 7. 主测试入口 ====================

if __name__ == "__main__":
    import math  # 为模拟器导入math
    
    print("开始测试机器人大脑系统...\n")
    
    # 运行主要测试
    test_robot_brain()
    
    # 测试其他功能
    test_task_updates()
    test_error_handling()
    
    print("\n✅ 所有测试完成！")