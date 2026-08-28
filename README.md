# 机械臂动力学参数辨识

本项目用于固定基座串联机械臂的激励轨迹生成、动力学仿真和参数辨识。当前主要面向 7 自由度 xArm7 与 nero，同时提供 xArm5、xArm6、xArm7 的独立 URDF 和 RViz 可视化文件。

可辨识的参数包括连杆质量、质心位置、转动惯量、库仑摩擦和黏性摩擦参数。

## 软件环境

推荐使用 Ubuntu 22.04、ROS 2 Humble 和 Python 3.10。主要 Python 依赖如下：

```bash
pip install numpy scipy matplotlib casadi pybullet mujoco \
  urdf-parser-py scikit-learn open3d xacro
```

项目还依赖 [OpTaS](https://github.com/cmower/optas)。物理一致性辨识使用 IPOPT；可选的 `dynid_core` 实现需要 Pinocchio 和 CVXPY。

ROS 可视化依赖：

```bash
sudo apt install ros-humble-rviz2 \
  ros-humble-robot-state-publisher \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui
```

## 目录结构

```text
dynamic_parameter_identification_ros/
├── src/
│   ├── traj_generation.py          # 优化并导出傅里叶激励轨迹
│   ├── mujoco_simulation.py        # MuJoCo 逆动力学仿真和力矩数据生成
│   ├── para_regression.py          # 动力学参数回归与结果保存
│   ├── dynamic_model.py            # 递归牛顿－欧拉模型和回归矩阵
│   └── dynid_core/                 # 可选的 ROS 无关 Pinocchio 辨识模块
└── urdf/                           # 旧模型文件

xarm_description/                   # xArm5/6/7 URDF、Mesh 和 RViz 文件
nero_description/                   # nero URDF、Mesh 和 RViz 文件
```

## 编译工作空间

在工作空间根目录执行：

```bash
cd ~/MyWorkspace/dynamic_parameter_identification_ros
source /opt/ros/humble/setup.bash
conda activate trajgen
colcon build --symlink-install
source install/setup.bash
```

每次重新编译后都需要重新执行 `source install/setup.bash`。

## URDF 可视化

xArm：

```bash
ros2 launch xarm_description xarm5_rviz_display.launch.py
ros2 launch xarm_description xarm6_rviz_display.launch.py
ros2 launch xarm_description xarm7_rviz_display.launch.py
```

nero：

```bash
ros2 launch nero_description display_urdf.launch.py
```

## 推荐运行流程

### 1. 生成激励轨迹

```bash
python3 dynamic_parameter_identification_ros/src/traj_generation.py
```

当前入口默认读取：

```text
xarm_description/urdf/xarm7_description.urdf
```

生成的轨迹保存到：

```text
/tmp/target_joint_states.csv
```

更换机械臂时，需要同步修改 `mainO()` 中的 URDF、末端连杆名称、关节零偏 `bias` 和关节范围 `q_min`、`q_max`。关节零偏必须加入最终导出的轨迹，不能只用于优化约束。

当前碰撞约束处于关闭状态。生成结果只保证显式加入优化器的关节位置、速度和加速度等约束；用于实机前必须另外进行自碰撞、环境碰撞和关节限位检查。

### 2. 生成 MuJoCo 动力学数据

```bash
python3 dynamic_parameter_identification_ros/src/mujoco_simulation.py
```

程序读取 `/tmp/target_joint_states.csv`，通过数值微分计算关节速度和加速度，再调用 MuJoCo 逆动力学得到关节力矩。默认输出文件为：

```text
dynamic_parameter_identification_ros/src/test_data/mujoco_robot_data.csv
```

无图形界面运行时，可以在代码中调用：

```python
simulator.run_sim(output_path, use_gui=False)
```

MuJoCo 的接触模型会影响正向动力学仿真，但当前脚本的主要输出来自 `mj_inverse()`。如果轨迹姿态发生接触，应单独检查接触力和约束力，避免将其误当作纯机械臂动力学力矩。

也可以不提供轨迹，直接在 URDF 关节范围内随机采样静态姿态并生成重力力矩数据：

```python
from mujoco_simulation import MuJoCoTrajectorySim

simulator = MuJoCoTrajectorySim(urdf_path)
rows = simulator.sample_static_joint_space(
    sample_count=1000,
    output_csv="static_robot_data.csv",
    seed=0,
)
```

静态采样时所有关节速度和加速度均设为零。无限位关节默认在 `[-π, π]` 内采样；也可以通过 `joint_lower` 和 `joint_upper` 指定范围。默认关闭接触约束，防止随机自碰撞产生的接触力混入重力力矩。

### 3. 辨识动力学参数

```bash
python3 dynamic_parameter_identification_ros/src/para_regression.py
```

默认流程包括：

1. 读取 MuJoCo 生成的位置和力矩数据；
2. 对速度和力矩进行低通滤波；
3. 构造动力学回归矩阵；
4. 使用带物理一致性约束的优化器估计参数；
5. 保存参数并比较预测力矩与输入力矩。

辨识结果保存到：

```text
dynamic_parameter_identification_ros/src/test_data/DynamicParameters.csv
```

文件每行对应动力学链中的一个刚体，行数根据 URDF 自动确定；无额外末端刚体时会在内部使用零参数虚拟末端。列顺序为质量、质心和惯量矩阵元素，数值保留 5 位小数。

主流程会从 URDF 自动识别唯一的叶子连杆、活动关节数和动力学参数维度，因此同一套代码可以处理 xArm5、xArm6、xArm7 和 nero。若 URDF 存在多个分支末端，需要在创建 `TrajGenerationUsrPath` 或 `Estimator` 时显式传入 `ee_link`。

## 如何评价辨识结果

辨识得到的单个质量、质心或惯量参数不一定与 URDF 完全一致。机械臂的完整动力学参数通常存在不可辨识的线性组合，因此应重点检查：

- 回归矩阵的秩和条件数；
- 预测力矩与测量力矩的 RMSE；
- 未参与拟合的验证轨迹上的预测误差；
- 质量和惯量是否满足物理一致性；
- 摩擦、负载、传感器零偏和采样时间是否建模正确。

当 IPOPT 出现 `Maximum_Iterations_Exceeded` 或 `Infeasible_Problem_Detected` 时，不应只增加最大迭代次数。应优先检查输入数据量级、回归矩阵缩放、初值、关节范围、约束冲突以及速度和加速度中的差分噪声。

## 可选的 `dynid_core`

`dynid_core` 提供一套不启动 ROS 的 Pinocchio 辨识接口：

```bash
cd dynamic_parameter_identification_ros
PYTHONPATH=src python3 -m dynid_core.cli \
  /path/to/robot.urdf /path/to/measurements.csv --dt 0.01
```

添加 `--physical` 可启用基于 CVXPY 伪惯性矩阵 LMI 的物理一致性约束。该模块目前是可选实现，尚未替代 `para_regression.py` 的默认流程。

## 注意事项

- 当前主要代码按固定基座、旋转关节机械臂设计。
- 轨迹 CSV 的采样周期必须与 MuJoCo `timestep` 和数据滤波参数一致。
- `Xlib: extension "NV-GLX" missing` 表示显示环境没有可用的 NVIDIA GLX；可以使用 `use_gui=False` 无界面运行。
- 对耗时较长的 IPOPT 优化，可以设置 `max_cpu_time` 和合理的 `max_iter`，并正确处理 `KeyboardInterrupt`。
- 实机运行前必须增加速度、加速度、力矩、碰撞和急停保护。

## 许可证

MIT
