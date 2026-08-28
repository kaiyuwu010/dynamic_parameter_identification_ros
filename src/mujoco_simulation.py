#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import xacro
from ament_index_python.packages import get_package_share_directory


def _resolve_package_uri(uri: str) -> str:
    # 不是以package://开头返回
    if not uri.startswith("package://"):
        return uri
    # 把package://后面的字符串，从/处划分为两部分
    package_and_path = uri[len("package://") :].split("/", 1)
    # 第一部分为包名
    package = package_and_path[0]
    # 第二部分是相对路径
    relative_path = package_and_path[1] if len(package_and_path) == 2 else ""
    # 获得完整路径
    return os.path.join(get_package_share_directory(package), relative_path)

# 把轨迹转换为形状(采样点数, 自由度数)
def _load_trajectory(trajectory, dof: int) -> np.ndarray:
    if isinstance(trajectory, (str, os.PathLike)):
        # 按照csv读取文件
        with open(trajectory, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        # 保存2到8列，第一列为时间戳
        values = np.asarray([[float(value) for value in list(row.values())[1 : dof + 1]] for row in rows], dtype=float,)
    else:
        # 输入是列表或NumPy数组，读取2到8列，第一列删除
        values = np.asarray(trajectory, dtype=float)
        if values.ndim == 2 and values.shape[1] == dof + 1:
            values = values[:, 1:]
    # 维度是否为2，列数是否等于自由度，行数(轨迹点数)是否小于2
    if values.ndim != 2 or values.shape[1] != dof or values.shape[0] < 2:
        raise ValueError(f"轨迹形状必须是 (采样点数, {dof})")
    if not np.all(np.isfinite(values)):
        raise ValueError("轨迹包含nan值或无穷")
    return values

# 加载urdf/xacro并用mujoco执行关节轨迹
class MuJoCoTrajectorySim:
    def __init__(self, model_path, trajectory, *, timestep: float = 0.01, gravity=(0.0, 0.0, -9.81),):
        if timestep <= 0.0:
            raise ValueError("timestep must be positive")
        self.timestep = float(timestep)
        self.model = self._load_model(model_path)
        self.model.opt.timestep = self.timestep
        self.model.opt.gravity[:] = np.asarray(gravity, dtype=float)
        self.data = mujoco.MjData(self.model)
        # 指定关节类型: 旋转关节、移动关节
        movable_types = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        # 找出活动关节编号
        self.joint_ids = [joint_id for joint_id in range(self.model.njnt) if int(self.model.jnt_type[joint_id]) in movable_types]
        if not self.joint_ids:
            raise ValueError("model contains no movable scalar joints")
        # 获取关节位置索引
        self.qpos_indices = np.asarray([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=int,)
        # 获取关节速度/力矩索引
        self.dof_indices = np.asarray([self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=int,)
        # 加载关节轨迹
        self.trajectory = _load_trajectory(trajectory, len(self.joint_ids))

    @staticmethod
    def _load_model(model_path) -> mujoco.MjModel:
        # 规范化模型路径
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"没有找到模型: {model_path}")
        # 如果是 Xacro，先展开宏和 include
        if model_path.name.endswith(".xacro"):
            xml_text = xacro.process_file(str(model_path)).toxml()
        else:
            xml_text = model_path.read_text(encoding="utf-8")
        # 解析XML
        root = ET.fromstring(xml_text)
        # 把mesh的package://路径替换为绝对路径
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename")
            if filename:
                mesh.set("filename", _resolve_package_uri(filename))
        # 将处理后的URDF写入临时目录
        with tempfile.TemporaryDirectory(prefix="mujoco_urdf_") as directory:
            resolved_urdf = Path(directory) / "resolved.urdf"
            ET.ElementTree(root).write(resolved_urdf, encoding="utf-8", xml_declaration=True,)
            return mujoco.MjModel.from_xml_path(str(resolved_urdf))

    # 仿真轨迹得到力矩
    def run_sim(self, output_csv=None, use_gui=True) -> list[list[float]]:
        q = self.trajectory                                                  # 关节位置形状(采样点数，关节数)
        edge_order = 2 if q.shape[0] >= 3 else 1                             # 采样点数大于3，采样二阶精度求导，只有两个点，只能采用一阶精度求导
        qd = np.gradient(q, self.timestep, axis=0, edge_order=edge_order)    # 数值微分得到关节速度
        qdd = np.gradient(qd, self.timestep, axis=0, edge_order=edge_order)  # 数值微分得到关节加速度
        # rows保存位置和力矩对
        rows = []
        # 启动可视化界面
        viewer = mujoco.viewer.launch_passive(self.model, self.data) if use_gui else None
        # 将每个时刻的状态写入MuJoCo数据结构
        for position, velocity, acceleration in zip(q, qd, qdd):
            self.data.qpos[self.qpos_indices] = position
            self.data.qvel[self.dof_indices] = velocity
            self.data.qacc[self.dof_indices] = acceleration
            # 计算逆动力学
            mujoco.mj_inverse(self.model, self.data)
            torque = self.data.qfrc_inverse[self.dof_indices].copy()
            mujoco.mj_forward(self.model, self.data)
            rows.append(np.concatenate((position, torque)).tolist())
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(self.timestep)
        # 关闭可视化界面
        if viewer is not None:
            viewer.close()
        # 保存到csv文件
        if output_csv is not None:
            output_csv = Path(output_csv).expanduser().resolve()
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            dof = len(self.joint_ids)
            header = [f"Joint{i + 1}_Pos" for i in range(dof)]
            header += [f"Joint{i + 1}_Torque" for i in range(dof)]
            with output_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
        return rows


if __name__ == "__main__":
    package_dir = get_package_share_directory("xarm_description")
    # package_dir = get_package_share_directory("nero_description")
    urdf_path = os.path.join(package_dir, "urdf", "xarm7_description.urdf")
    # urdf_path = os.path.join(package_dir, "urdf", "nero_description.urdf")
    trajectory_path = "/tmp/target_joint_states.csv"
    project_dir = Path(__file__).resolve().parent.parent
    output_path = project_dir / "src" / "test_data" / "mujoco_robot_data.csv"
    simulator = MuJoCoTrajectorySim(urdf_path, trajectory_path, timestep=0.01)
    simulator.run_sim(output_path)
