#!/bin/bash

# =========================================================
# 一键启动脚本 (Ranger + Livox + FAST_LIO + Nav2 + TrackReturn)
# =========================================================

# 1. 提前获取 sudo 权限，避免中途弹密码打断自动化流程
echo "[1/6] 正在请求 sudo 权限以设置 CAN 接口..."
sudo -v

# 2. 设置一键退出机制 (极其重要)
# 捕获 Ctrl+C (SIGINT) 等信号，确保脚本退出时把所有后台的 ROS 节点一起干掉
trap 'echo -e "\n[INFO] 接收到退出信号，正在关闭所有 ROS 2 节点..."; kill $(jobs -p); exit 0' SIGINT SIGTERM EXIT


echo "=== Ensure CAN ready ==="

if ip -details link show can0 > /dev/null 2>&1; then
  if ip -details link show can0 | grep -q "UP"; then
    echo "[INFO] can0 already UP"
  else
    echo "[WARN] can0 exists but not usable, reinitializing..."

    # 删除旧接口
    sudo ip link delete can0 || true

    # 重新 bringup
    cd ~/agilex_ws
    sudo bash ~/agilex_ws/src/ranger_ros2/ranger_bringup/scripts/bringup_can2usb.bash
  fi
else
  echo "[INFO] can0 not found, running bringup..."
  cd ~/agilex_ws
  sudo bash ~/agilex_ws/src/ranger_ros2/ranger_bringup/scripts/bringup_can2usb.bash
fi

# 3. 启动底盘节点
echo "[2/6] 启动 Ranger 底盘节点..."
source ~/agilex_ws/install/setup.bash
ros2 launch ranger_bringup ranger_mini_v3.launch.py &
sleep 2  # 给底盘预留一点初始化时间

# 4. 启动雷达 (后台运行)
echo "[3/6] 启动 Livox MID360 雷达..."
source ~/workspace/install/setup.bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py &
sleep 2  # 给雷达预留初始化时间，防止点云话题丢失

# 5. 启动 SLAM/建图框架 (后台运行)
echo "[4/6] 启动 FAST-LIO-SAM 及其网格地图节点..."
ros2 launch fast_lio_sam_g gridmap_mid360.launch.py &

# 6. 等待 SLAM 稳定后启动 Nav2
echo "等待 SLAM 框架稳定 (5秒)..."
sleep 5
echo "[5/6] 启动 Nav2 导航栈..."
ros2 launch fast_lio_sam_g nav2_mid360.launch.py &
sleep 3

# 7. 启动轨迹记录 / 回放 / 定点导航管理节点
echo "[6/6] 启动 Track Return 轨迹管理节点..."
ros2 launch fast_lio_sam_g track_return_mid360.launch.py &

echo "========================================================="
echo "[SUCCESS] 所有系统已在后台成功启动！"
echo "[INFO] 现已支持以下上位机话题："
echo "       /goto_point   -> 去目标点，并自动开始记录本轮路径"
echo "       /repeat_path  -> 1 正向重走, -1 反向返回, 0 停止"
echo "       /Status       -> 状态反馈"
echo "[TIP] 请勿关闭此终端。按 【Ctrl + C】 即可一键安全停止所有节点。"
echo "========================================================="

# 7. 挂起主线程，保持脚本运行状态
wait
