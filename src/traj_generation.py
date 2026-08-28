#!/usr/bin/python3
import optas
import sys
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
from time import sleep, time, perf_counter, time_ns
from scipy.spatial.transform import Rotation as Rot
from optas.spatialmath import *
import os

import rclpy
import xacro
from ament_index_python import get_package_share_directory
from rclpy import qos
from rclpy.node import Node
import csv

import pathlib
import urdf_parser_py.urdf as urdf
import math
import copy
import random
import open3d as o3d
from sklearn.mixture import GaussianMixture

import casadi as cs
from dynamic_model import find_dyn_parm_deps, RNEA_function, DynamicLinearlization, getJointParametersfromURDF

# 检查数值或CasADi计算结果中是否包含NaN
def contains_nan(x):
    for i,element in enumerate(np.array(x).flatten()):
        if np.isnan(element):
            print("NAN occured on ", i)
            return True
    return False

# 将二维列表保存为CSV文件
def save_to_csv(values_list, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(values_list)
    print(f"Data saved to {filename}")

# 从不带表头的CSV中读取浮点数二维列表
def load_from_csv(filename):
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        loaded_list = [list(map(float, row)) for row in reader]
    return loaded_list

# 在关节空间中建立避碰约束
def getConstraintsinJointSpace(robot, point_coord = [0.]*3, Nb=7, 
                               base_link="link_3", 
                               base_joint_name="A3", 
                               ee_link="link_ee"):
    # 定义符号关节变量
    q = cs.SX.sym('q', Nb, 1)
    # 计算正运动学获取末端位姿
    pe = robot.get_global_link_position(ee_link, q)
    Re = robot.get_global_link_rotation(ee_link, q)
    # 获取待避碰连杆位姿
    pb = robot.get_global_link_position(base_link, q)
    Rb = robot.get_global_link_rotation(base_link, q)
    # 将末端凸包点变换到基坐标系
    pp = pe + Re[:,0]*point_coord[0] + Re[:,1]*point_coord[1] + Re[:,2]*point_coord[2]
    # 获取待避碰连杆的关节原点作为椭球中心
    robot_urdf = robot.urdf
    joint = robot_urdf.joint_map[base_joint_name]
    # 获取目标关节坐标系原点在父关节坐标系的位置向量
    xyz, _ = robot.get_joint_origin(joint)
    print("robot_urdf.joint_map = ",robot_urdf.joint_map)
    # 再把凸包点表示到待避碰连杆的局部坐标系中
    p = Rb.T@(pp -pb)
    x= p[0]
    y= p[1]
    z= p[2]
    # 椭球中心沿z轴偏移一半，避免碰撞约束过于严格
    c = xyz[2]
    c = c/2
    if(c==0):
        c = 0.2
    # 椭球长轴和短轴
    a = 0.15
    b = 0.15
    # E=0 表示凸包点位于椭球表面; E>0 表示凸包点位于椭球外部
    EpVF = x*x/(a*a) + y*y/(b*b) + (z-c)*(z-c)/(c*c) - 1
    # 将符号表达式封装为CasADi函数，便于优化器调用
    p_fun = optas.Function('A_fun', [q], [EpVF])
    return p_fun 

class FourierSeries():
    # 保存Fourier系数的阶数Rank、关节数channel、偏置bias和基频ff
    def __init__(self, Rank = 5, channel = 7, bias=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ff = 0.01) -> None:
        assert channel == len(bias), "Different size of Rank and bias!"
        # self.a = cs.SX.sym('a', Rank,channel)
        # self.b = cs.SX.sym('b', Rank,channel)
        self.Rank = Rank
        self.bias = bias
        self.ff = ff  # 单位hz
        self.channel = channel
    # 根据给定的时间t, Fourier系数a、b, 函数名name, 创建关节位置的符号函数
    def FourierFunction(self, t, a, b, name):
        q = copy.deepcopy(self.bias)
        for i in range(self.channel):
            for l in range(self.Rank):
                # 分别计算1、2、...5倍基频
                wl = ((l+1) * self.ff * math.pi * 2.0) 
                q[i] = q[i] + a[l,i]/wl * cs.sin(wl * t) - b[l,i]/wl * cs.cos(wl * t)
        return cs.Function(name, [a, b, t], q)
    # 根据给定的时间t, Fourier系数a、b, 计算关节位置的数值值, 不建立符号函数
    def FourierValue(self, a, b, t):
        q = copy.deepcopy(self.bias)
        for i in range(self.channel):
            for l in range(self.Rank):
                # 分别计算1、2、...5倍基频
                wl = ((l+1) * self.ff * math.pi * 2.0)  # wl单位是rad/s
                q[i] = q[i] + a[l,i]/wl * np.sin(wl * t) - b[l,i]/wl * np.cos(wl * t)
        return q

class TrajGeneration(Node):
    def __init__(self, node_name = "para_estimatior", dt_ = 5.0, N_ = 100, gravity_vector=[0, 0, -9.81]) -> None:
        super().__init__(node_name=node_name)
        self.declare_parameter("model", "med7")
        self.model_ = str(self.get_parameter("model").value)
        path = os.path.join(get_package_share_directory("lbr_description"), "urdf", self.model_, f"{self.model_}.urdf.xacro", )
        gv = gravity_vector
        self.initial_model_params(path, gv)
    # 根据URDF文件路径和重力向量初始化机器人模型、动力学函数和回归矩阵
    def initial_model_params(self, path, gv, ee_link = "link7"):
        self.robot = optas.RobotModel(xacro_filename = path, time_derivs=[1],)
        # 从URDF中提取关节参数
        Nb, xyzs, rpys, axes = getJointParametersfromURDF(self.robot, ee_link)
        # 构造逆动力学函数
        self.dynamics_ = RNEA_function(Nb, 1, rpys, xyzs, axes, gravity_para=cs.DM(gv))
        # 构造回归矩阵和完整动力学参数向量
        self.Ymat, self.PIvector = DynamicLinearlization(self.dynamics_, Nb)
        # 从URDF中读取名义惯性参数
        urdf_string_ = xacro.process(path)
        robot = urdf.URDF.from_xml_string(urdf_string_)
        # 保存各连杆的质量、质心位置和惯性矩阵
        masses = [link.inertial.mass for link in robot.links if link.inertial is not None]
        self.masses_np = np.array(masses[1:])
        massesCenter = [link.inertial.origin.xyz for link in robot.links if link.inertial is not None]
        self.massesCenter_np = np.array(massesCenter[1:]).T
        Inertia = [link.inertial.inertia.to_matrix() for link in robot.links if link.inertial is not None]
        self.Inertia_np = np.hstack(tuple(Inertia[1:]))
        
    # 读取带表头CSV，忽略列名并按文件中的列顺序返回浮点行
    @ staticmethod
    def readCsvToList(path):
        l = []
        with open(path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                joint_names = [x.strip() for x in list(row.keys())]
                l.append([float(x) for x in row.values()])
        return l    
    
    # 读取测量文件，并用一阶差分估计关节速度
    def ExtractFromMeasurmentCsv(self):
        path_pos = os.path.join(get_package_share_directory("gravity_compensation"), "test", "measurements_with_ext_tau.csv",)
        # 采样时间间隔
        dt = 0.01
        pos_l = []
        tau_ext_l = []
        with open(path_pos) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # 前7列关节位置
                pl = list(row.values())[0:7]
                # 后7列外部力矩
                tl = list(row.values())[7:14]
                # 关节位置转换为浮点数保存到列表中
                pos_l.append([float(x) for x in pl])
                # 外部力矩转换为浮点数保存到列表中
                tau_ext_l.append([float(x) for x in tl])
        vel_l =[]
        for id in range(len(pos_l)):
            if id == 0:
                vel_l.append([0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0])
            else:
                # 用一阶差分估计关节速度
                vel_l.append([(p-p_1)/dt for (p, p_1) in zip(pos_l[id], pos_l[id-1])])
        return pos_l, vel_l, tau_ext_l
    
    # 生成优化轨迹，返回Fourier系数a、b和信息矩阵函数fc
    def generate_opt_traj_Link(self, Ff,                                              # 傅立叶轨迹基频，表示每秒几个周期，单位hz
                               sampling_rate,                                         # 采样频率，表示每秒采集多少点，单位hz
                               Rank=5,                                                # 傅立叶谐波阶次
                               q_min = [-6.2, -6.2, -6.2, -6.2, -6.2, -6.2, -6.2],    # 关节范围下限，单位rad
                               q_max = [ 6.2,  6.2,  6.2,  6.2,  6.2,  6.2,  6.2],    # 关节范围上限，单位rad 
                               q_vmin = [-6.2, -6.2, -6.2, -6.2, -6.2, -6.2, -6.2],   # 关节速度下限，单位rad/s
                               q_vmax = [ 6.2,  6.2,  6.2,  6.2,  6.2,  6.2,  6.2],   # 关节速度上限，单位rad/s
                               f_path = None, g_path=None, 
                               bias = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        # 输入检查
        if Ff <= 0.0 or sampling_rate <= 0.0:
            raise ValueError("基频和采样率必须为正")
        if Rank < 2:
            raise ValueError("谐波阶数至少为 2")
        q_min = np.asarray(q_min, dtype=float)
        q_max = np.asarray(q_max, dtype=float)
        q_vmin = np.asarray(q_vmin, dtype=float)
        q_vmax = np.asarray(q_vmax, dtype=float)
        bias = np.asarray(bias, dtype=float)
        for name, values in (("q_min", q_min), ("q_max", q_max), ("q_vmin", q_vmin), ("q_vmax", q_vmax), ("bias", bias),):
            if values.shape != (7,):
                raise ValueError(f"{name} 必须包含 7 个值")
        # 位置幅值约束
        position_margins = np.minimum(q_max - bias, bias - q_min)
        # 速度幅值约束
        velocity_margins = np.minimum(q_vmax, -q_vmin)
        if np.any(position_margins <= 0.0):
            raise ValueError("bias must lie strictly inside every joint position bound")
        if np.any(velocity_margins <= 0.0):
            raise ValueError("joint velocity bounds must contain zero")
        # 单周期采样点数
        pointsNum = int(sampling_rate / Ff)
        if pointsNum < 1:
            raise ValueError("sampling_rate / Ff must be at least 1")
        print("一个周期内的样本数:", pointsNum)
        # 初始化傅立叶系数变量
        x = cs.MX.sym('x', 2 * Rank * 7, 1)
        ab = cs.reshape(x, 2 * Rank, 7)
        a = ab[:Rank, :]
        b = ab[Rank:, :]
        # 求最小参数集提取矩阵
        Pb, _, _ = find_dyn_parm_deps(7, 80, self.Ymat)
        Pb = cs.DM(Pb)
        bias_dm = cs.reshape(cs.DM(bias), 7, 1)
        # 当前按你的要求关闭碰撞约束。将这里替换为非空点集即可恢复避碰项。
        points = np.empty((0, 3))
        print("凸包点数:", len(points))
        vfs_fun = []
        for point in points:
            for link_index in range(2, 6):
                vfs_fun.append(getConstraintsinJointSpace(self.robot, point_coord=point, base_link="link_" + str(link_index), base_joint_name="A" + str(link_index),))
        Y_blocks = []  # 最小参数集对应的回归矩阵
        pfun_list = [] # 各个采样点的碰撞约束
        for sample_index in range(pointsNum):
            tc = sample_index / sampling_rate
            q = bias_dm           # 从零偏开始叠加傅立叶轨迹
            qd = cs.MX.zeros(7, 1)
            qdd = cs.MX.zeros(7, 1)
            for harmonic_index in range(Rank):
                wl = (harmonic_index + 1) * 2.0 * math.pi * Ff
                a_l = a[harmonic_index, :].T
                b_l = b[harmonic_index, :].T
                # 构造位置、速度、加速度符号向量
                q = q + a_l / wl * math.sin(wl * tc) - b_l / wl * math.cos(wl * tc)
                qd = qd + a_l * math.cos(wl * tc) + b_l * math.sin(wl * tc)
                qdd = qdd - a_l * wl * math.sin(wl * tc) + b_l * wl * math.cos(wl * tc)
            Y_blocks.append(self.Ymat(q, qd, qdd) @ Pb)
            for vf_fun in vfs_fun:
                pfun_list.append(vf_fun(q))
        Y = cs.vertcat(*Y_blocks)
        # 保留原有的三组线性周期条件及位置/速度幅值约束，只把表达式类型改为MX
        a_eq1 = [0.0] * 7
        a_eq2 = [0.0] * 7
        b_eq1 = [0.0] * 7
        position_amplitude = [0.0] * 7
        velocity_amplitude = [0.0] * 7
        for joint_index in range(7):
            for harmonic_index in range(Rank):
                order = harmonic_index + 1
                wl = order * 2.0 * math.pi * Ff
                a_value = a[harmonic_index, joint_index]
                b_value = b[harmonic_index, joint_index]
                a_eq1[joint_index] += b_value / order                                   # t=0时的位置，q(0) = 0 
                a_eq2[joint_index] += a_value                                           # t=0时的速度，dotq(0) = 0
                b_eq1[joint_index] += b_value * order                                   # t=0时的加速度，ddotq(0) = 0 
                amplitude = cs.sqrt(a_value * a_value + b_value * b_value + 1e-12)      # asinx - bcosx = sqrt{a^2+b^2}sin(x-&)，所以幅值最大为a^2+b^2
                position_amplitude[joint_index] += amplitude / wl                       # 位置幅值
                velocity_amplitude[joint_index] += amplitude                            # 速度幅值
        # 约束和上下界
        g = cs.vertcat(*(a_eq1 + a_eq2 + b_eq1 + position_amplitude + velocity_amplitude + pfun_list))
        lbg = cs.DM([0.0] * (35 + len(pfun_list)))                                                                   # 矩阵形状(35 + n, 1)
        ubg = cs.DM([0.0] * 21 + position_margins.tolist() + velocity_margins.tolist() + [1e10] * len(pfun_list))    # 矩阵形状(21 + p + v + n, 1)
        # A_reg必须正定
        A = Y.T @ Y
        identity = cs.DM.eye(A.size1())
        # 根据 A 自身尺度设置正则化，避免固定 1e-6 太大或太小
        mean_diagonal = cs.trace(A) / A.size1()
        regularization = 1e-6 * (mean_diagonal + 1.0)
        A_reg = A + regularization * identity
        # regularization = 1e-6
        # A_reg = A + regularization * cs.DM.eye(A.size1())
        # A-optimal：最小化参数估计方差
        f = (cs.trace(cs.solve(A_reg, identity)) * mean_diagonal / A.size1())
        # f = cs.norm_fro(A_reg) * cs.norm_fro(cs.solve(A_reg, cs.DM.eye(A.size1()))) # cs.solve比cs.inv更稳定
        Y_x_fun = cs.Function('Y_x_fun', [x], [Y])          # 把计算回归矩阵Y的符号表达式封装为函数，输入傅立叶系数x
        g_x_fun = cs.Function('g_x_fun', [x], [g])          # 把计算约束g的符号表达式封装为函数，输入傅立叶系数x
        A_x_fun = cs.Function('A_x_fun', [x], [A_reg])      # 把计算信息矩阵A_reg的符号表达式封装为函数，输入傅立叶系数x
        a_eval = cs.MX.sym('a_eval', Rank, 7)
        b_eval = cs.MX.sym('b_eval', Rank, 7)
        x_eval = cs.vec(cs.vertcat(a_eval, b_eval))
        Y_fun = cs.Function('Y_fun', [a_eval, b_eval], [Y_x_fun(x_eval)])  # 把计算回归矩阵Y的符号表达式封装为函数，输入傅立叶系数a、b
        fc = cs.Function('fc', [a_eval, b_eval], [A_x_fun(x_eval)])        # 把计算信息矩阵A_reg的符号表达式封装为函数，输入傅立叶系数a、b
        # 求解问题
        problem = {'x': x, 'f': f, 'g': g}
        solver_options = {'expand': False, 
                          'verbose': False, 
                          'ipopt': {'max_iter': 1000, 
                                    "tol": 1e-6,
                                    "acceptable_tol": 1e-4,
                                    "acceptable_iter": 15,
                                    "acceptable_dual_inf_tol": 1e-2,
                                    "constr_viol_tol": 1e-3,
                                    "hessian_approximation": "limited-memory",
                                    "limited_memory_max_history": 20,
                                    "mu_strategy": "adaptive",
                                    "nlp_scaling_method": "gradient-based",
                                    "bound_relax_factor": 1e-8,
                                    "linear_solver": "mumps",
                                    "print_level": 5,
                                   },}
        S = cs.nlpsol('S', 'ipopt', problem, solver_options)
        print("MX 函数节点数: Y = {}, g = {}".format(Y_x_fun.n_nodes(), g_x_fun.n_nodes()))
        # 生成满足三组线性等式且位于幅值边界内的非零初值
        def make_initial_guess(rng, coefficient_scale=0.2):
            a0 = rng.uniform(-coefficient_scale, coefficient_scale, size=(Rank, 7))
            b0 = rng.uniform(-coefficient_scale, coefficient_scale, size=(Rank, 7))
            harmonic_orders = np.arange(1, Rank + 1, dtype=float)
            a0 -= np.mean(a0, axis=0, keepdims=True)
            constraints_b = np.vstack((1.0 / harmonic_orders, harmonic_orders))
            projection_b = constraints_b.T @ np.linalg.pinv(constraints_b @ constraints_b.T)
            b0 -= projection_b @ (constraints_b @ b0)
            angular_frequencies = 2.0 * math.pi * Ff * harmonic_orders
            position_sum = np.sum(np.hypot(a0, b0) / angular_frequencies[:, None], axis=0)
            velocity_sum = np.sum(np.hypot(a0, b0), axis=0)
            for joint_index in range(7):
                scale = 1.0
                if position_sum[joint_index] > 0.0:
                    scale = min(scale, 0.5 * position_margins[joint_index] / position_sum[joint_index])
                if velocity_sum[joint_index] > 0.0:
                    scale = min(scale, 0.5 * velocity_margins[joint_index] / velocity_sum[joint_index])
                a0[:, joint_index] *= scale
                b0[:, joint_index] *= scale
            # CasADi 的 reshape/vec 是列主序；NumPy 这里必须明确 order='F'。
            return np.vstack((a0, b0)).reshape((-1, 1), order='F')

        init_x0 = make_initial_guess(np.random.default_rng(0))
        initial_g = np.asarray(g_x_fun(init_x0).full(), dtype=float).reshape(-1)
        lbg_np = np.asarray(lbg.full(), dtype=float).reshape(-1)
        ubg_np = np.asarray(ubg.full(), dtype=float).reshape(-1)
        initial_violation = max(0.0, float(np.max(lbg_np - initial_g)), float(np.max(initial_g - ubg_np)),)
        print("初值最大约束违反:", initial_violation)
        if not np.isfinite(initial_violation) or initial_violation > 1e-9:
            raise RuntimeError("生成的初值不可行")
        sol = S(x0=init_x0, lbg=lbg, ubg=ubg)
        stats = S.stats()
        status = stats.get('return_status', 'unknown')
        print("IPOPT状态:", status)
        
        candidate_g = np.asarray(g_x_fun(sol["x"]).full(), dtype=float, ).reshape(-1)
        lbg_np = np.asarray(lbg.full(), dtype=float).reshape(-1)
        ubg_np = np.asarray(ubg.full(), dtype=float).reshape(-1)
        lower_violation = np.maximum(lbg_np - candidate_g, 0.0)
        upper_violation = np.maximum(candidate_g - ubg_np, 0.0)
        violations = np.maximum(lower_violation, upper_violation)
        names = (
            [f"position_eq_J{i + 1}" for i in range(7)]
            + [f"velocity_eq_J{i + 1}" for i in range(7)]
            + [f"acceleration_eq_J{i + 1}" for i in range(7)]
            + [f"position_amplitude_J{i + 1}" for i in range(7)]
            + [f"velocity_amplitude_J{i + 1}" for i in range(7)]
        )
        print("最大约束违反:")
        for index in np.argsort(violations)[::-1][:10]:
            print(
                f"{names[index]}: "
                f"value={candidate_g[index]:.8g}, "
                f"bounds=[{lbg_np[index]:.8g}, {ubg_np[index]:.8g}], "
                f"violation={violations[index]:.8g}"
            )         
        # if not stats.get('success', False):
        #     raise RuntimeError(f"IPOPT未成功收敛: {status}")
        ab_best = np.asarray(sol['x'].full(), dtype=float).reshape((2 * Rank, 7), order='F')
        a_best = ab_best[:Rank, :]
        b_best = ab_best[Rank:, :]
        final_g = np.asarray(g_x_fun(sol['x']).full(), dtype=float).reshape(-1)
        final_violation = max(0.0, float(np.max(lbg_np - final_g)), float(np.max(final_g - ubg_np)),)
        print("最终最大约束违反:", final_violation)
        # 此处打印的是数值 Y，而不是会导致内存爆炸的符号 SX 表达式。
        Y_numeric = np.asarray(Y_fun(a_best, b_best).full(), dtype=float)
        print("Y数值矩阵尺寸:", Y_numeric.shape)
        # with np.printoptions(precision=6, suppress=True, linewidth=240, threshold=np.inf):
            # print("Y数值矩阵:\n", Y_numeric)
        A_raw = Y_numeric.T @ Y_numeric
        A_raw = 0.5 * (A_raw + A_raw.T)
        raw_eigenvalues = np.linalg.eigvalsh(A_raw)
        A_reg_numeric = np.asarray(fc(a_best, b_best).full(), dtype=float)
        A_reg_numeric = 0.5 * (A_reg_numeric + A_reg_numeric.T)
        regularized_eigenvalues = np.linalg.eigvalsh(A_reg_numeric)
        tolerance = max(A_raw.shape) * np.finfo(float).eps * max(float(raw_eigenvalues[-1]), 1.0)
        numerical_rank = int(np.sum(raw_eigenvalues > tolerance))
        raw_condition_a = math.inf if raw_eigenvalues[0] <= tolerance else raw_eigenvalues[-1] / raw_eigenvalues[0]
        regularized_condition_a = regularized_eigenvalues[-1] / regularized_eigenvalues[0]
        print("原始信息矩阵特征值: \n", raw_eigenvalues)
        print("原始信息矩阵数值秩: {}/{}".format(numerical_rank, A_raw.shape[0]))
        print("cond(A) 原始/正则化:", raw_condition_a, regularized_condition_a)
        print("cond(Y) 原始/正则化:", math.sqrt(raw_condition_a), math.sqrt(regularized_condition_a))
        print("对应的傅立叶系数:\na = {}\nb = {}".format(a_best, b_best))
        return a_best, b_best, fc
    
    def get_ineq_Fourier_expression(self, Ff, a, b, q_min, q_max, q_vmin, q_vmax, ):
        """建立 Fourier 系数对应的周期边界及位置、速度幅值约束。

        返回 ``g, lbg, ubg``，求解器统一按 ``lbg <= g <= ubg`` 处理。
        当前内部谐波循环固定为 5，因此 Rank 不为 5 时需要同步泛化。

        约束向量按以下顺序拼接：
            a_eq1: sum(a_l/(l+1)) = 0，约束周期端点的位置关系。
            a_eq2: sum((l+1)*a_l) = 0，约束周期端点的加速度关系。
            b_eq1: sum(b_l) = 0，约束周期端点的速度关系。
            ab_sq_ineq1: 各阶位置幅值之和，作为位置偏移的保守上界。
            ab_sq_ineq2: 各阶速度幅值之和，作为速度的保守上界。
            ab_sq_ineq3: 展平的 a_l、b_l，限制单个谐波系数。

        前三组的上下界都为 0，因此是等式约束；后三组为区间约束。
        """
        # 每个数组保存7个关节各自的符号约束表达式
        a_eq1 = [0.0]*7
        a_eq2 = [0.0]*7
        b_eq1 = [0.0]*7
        ab_sq_ineq1 = [0.0]*7
        ab_sq_ineq2 = [0.0]*7
        ab_sq_ineq3 = []
        # lbg*/ubg*与上述表达式分组一一对应，最后按相同顺序拼接。
        lbg1 = []
        lbg2 = []
        lbg3 = []
        lbg4 = []
        lbg5 = []
        lbg6 = []
        
        ubg1 = []
        ubg2 = []
        ubg3 = []
        ubg4 = []
        ubg5 = []
        ubg6 = []
        # 外层遍历关节，内层遍历5个Fourier谐波。
        for i in range(7):
            for l in range(5):
                # 这三个和式在周期边界处被强制为0
                a_eq1[i] = a_eq1[i] + a[l,i]/(l+1)
                b_eq1[i] = b_eq1[i] + b[l,i]
                a_eq2[i] = a_eq2[i] + a[l,i]*(l+1)
                wl = ((l+1) * Ff* math.pi* 2.0) 
                # sqrt(a^2+b^2)/w 是该阶谐波的位置振幅
                ab_sq_ineq1[i] = (ab_sq_ineq1[i] + 1.0/(wl)* cs.sqrt(a[l,i]*a[l,i] + b[l,i]*b[l,i]))
                # q 对时间求导后 1/w 被抵消，sqrt(a^2+b^2) 是速度振幅。
                ab_sq_ineq2[i] = (ab_sq_ineq2[i]+ 
                cs.sqrt(a[l,i]*a[l,i] + b[l,i]*b[l,i]))
                ab_sq_ineq3.append(a[l,i])
                ab_sq_ineq3.append(b[l,i])
                # 单系数边界取位置诱导界和速度界中更严格的一项
                cpr2 = min((l+1)*Ff/5.0*2.0*math.pi*q_max[i], q_vmax[i])
                cpr = max((l+1)*Ff/5.0*2.0*math.pi*q_min[i], q_vmin[i])
                lbg6.append(cpr)
                lbg6.append(cpr)
                ubg6.append(cpr2)
                ubg6.append(cpr2)
            # 前三组上下界相等，所以IPOPT将其视为等式约束
            lbg1.append(0.0)
            lbg2.append(0.0)
            lbg3.append(0.0)
            lbg4.append(0.0)
            lbg5.append(0.0)
            ubg1.append(0.0)
            ubg2.append(0.0)
            ubg3.append(0.0)
            ubg4.append(Ff * math.pi * 2.0 * q_max[i])
            ubg5.append(q_vmax[i])
        # CasADi NLP只接受一个g，表达式和上下界必须保持完全相同的顺序
        g = cs.vertcat(*(a_eq1 +  a_eq2 +  b_eq1 +  ab_sq_ineq1 + ab_sq_ineq2 + ab_sq_ineq3))
        lbg = cs.vertcat(*(lbg1, lbg2, lbg3, lbg4, lbg5, lbg6))
        ubg = cs.vertcat(*(ubg1, ubg2, ubg3, ubg4, ubg5, ubg6))
        return g, lbg, ubg
    
    # 堆叠各采样时刻的基本惯性参数与摩擦参数回归矩阵
    def get_Y_matrix(self, q_list, qd_list, qdd_list):
        """
        惯性部分为 ``Y(q, qd, qdd) @ Pb``；摩擦部分由库仑项
        ``diag(sign(qd))`` 和黏性项 ``diag(qd)`` 组成。

        Args:
            q_list: N 个 n 维关节位置列向量。
            qd_list: 与 q_list 对齐的关节速度列向量。
            qdd_list: 与 q_list 对齐的关节加速度列向量。

        Returns:
            堆叠矩阵 Y。若基本惯性参数数为 p_b，则形状为
            ``(N*n, p_b+2*n)``；最后 2*n 列分别表示库仑和黏性摩擦。
        """
        
        # Pb 把完整惯性参数投影到可辨识的最小参数集合。
        Pb, _, __ =find_dyn_parm_deps(self.robot.ndof, 80, self.Ymat)
        Y_ = []
        Y_fri = []
        for q, qd, qdd in zip(q_list, qd_list, qdd_list):
            # 单时刻刚体回归块有 n 行、p_b 列。
            Y_temp = self.Ymat(q, qd, qdd) @Pb
            fri_ = cs.diag(cs.sign(qd))
            fri_ = cs.horzcat(fri_,  cs.diag(qd))
            Y_.append(Y_temp)
            Y_fri.append(fri_)
        # 沿行方向堆叠后，两个矩阵都具有 N*n 行。
        Y1= optas.vertcat(*Y_)
        Y_fri1 = optas.vertcat(*Y_fri)
        # 水平拼接，使同一行同时描述刚体动力学与摩擦力矩。
        Y = optas.horzcat(Y1, Y_fri1)
        return Y
    
    # 获取优化问题的 CasADi/IPOPT 求解器、约束上下界和信息矩阵函数
    def get_optimization_problem(self,Ff, sampling_rate, Rank=5, q_min=-3.0*np.ones(7), q_max =3.0*np.ones(7), q_vmin=-6.0*np.ones(7),q_vmax=6.0*np.ones(7), f_path = None, g_path=None,bias=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        """构建 Fourier 激励轨迹的非线性规划问题。

        优化变量为 a、b，共 ``2*Rank*7`` 个。目标通过改善信息矩阵
        ``A=Y.T@Y`` 的条件性来增强参数可辨识性；约束包含周期边界、
        关节位置/速度范围以及末端凸包点的近似自碰撞约束。

        Args:
            Ff: Fourier 基频 Hz，轨迹周期为 ``T=1/Ff``。
            sampling_rate: 构造优化问题时使用的离散采样频率 Hz。
            Rank: Fourier 阶数。
            q_min/q_max: 每个关节的位置下界和上界，单位 rad。
            q_vmin/q_vmax: 每个关节的速度下界和上界，单位 rad/s。
            f_path/g_path: 遗留参数，当前没有读取这两个路径。
            bias: 周期轨迹的关节中心位置，单位 rad。

        Returns:
            S: CasADi/IPOPT 求解器。
            lbg, ubg: 约束向量的下界和上界。
            fc: 给定 a、b 后计算信息矩阵 A 的函数。
        """
        # 定义Fourier正弦/余弦系数以及时间符号变量
        a = cs.SX.sym('a', Rank, 7)
        b = cs.SX.sym('b', Rank, 7)
        t = cs.SX.sym('t', 1)
        # 用符号变量构造q(t)
        fourierInstance = FourierSeries(ff = Ff,bias=bias)
        fourierF = fourierInstance.FourierFunction(t, a, b,'f1')
        fourier = fourierF(a,b,t)
        # 使用符号自动微分，求qd、qdd
        fourierDot = [optas.jacobian(fourier[i],t) for i in range(len(fourier))]
        fourierDDot = [optas.jacobian(fourierDot[i],t) for i in range(len(fourierDot))]
        # 一个完整周期T=1/Ff内，按sampling_rate均匀采样
        ts = [1.0/(sampling_rate)*k for k in range(int(sampling_rate/(Ff)))]
        # 三个列表的第k项严格对应同一个采样时刻ts[k]。
        q_list = []
        qd_list = []
        qdd_list = []
        for tc in ts:
            q_list.append(cs.vertcat(*[optas.substitute(id, t, tc) for id in fourier]))
            qd_list.append(cs.vertcat(*[optas.substitute(id, t, tc) for id in fourierDot]))
            qdd_list.append(cs.vertcat(*[optas.substitute(id, t, tc) for id in fourierDDot]))
        # 用末端网格凸包点与各连杆椭球的相对位置近似自碰撞约束。
        path_pos = os.path.join(get_package_share_directory("med7_dock_description"), "meshes", "EndEffector.STL", )
        # 凸包用于减少 STL 点数；当前网格路径及连杆名称仍是 MED7 专用
        mesh = o3d.io.read_triangle_mesh(path_pos)
        convex_hull, _ = mesh.compute_convex_hull()
        hull_points = np.asarray(convex_hull.vertices)
        scores = []
        for component_count in range(1, 11):
            model = GaussianMixture(n_components=component_count, random_state=0)
            model.fit(hull_points)
            scores.append(model.score(hull_points))
        best_component_count = int(np.argmax(scores)) + 1
        model = GaussianMixture(n_components=best_component_count, random_state=0)
        model.fit(hull_points)
        points = model.means_
        vfs_fun = []
        for point in points:
            for i in range(2,6):
                vfs_fun.append(getConstraintsinJointSpace(self.robot, point_coord=point, base_link="link_"+str(i), base_joint_name="A"+str(i)))     
        # 包含“时间点 × 凸包点 × 待避碰连杆”的所有标量避碰表达式
        pfun_list = []
        for q in q_list:
            for j in range(len(vfs_fun)):
                pfun_list.append(vfs_fun[j](q))
        # Y的列同时覆盖基本惯性参数、库仑摩擦和黏性摩擦
        Y = self.get_Y_matrix(q_list, qd_list, qdd_list)
        # 信息矩阵；越远离奇异，噪声对参数估计结果的放大越小
        A = Y.T @ Y
        A_inv = cs.inv(A)
        # 先建立Fourier周期、位置和速度约束
        g, lbg, ubg = self.get_ineq_Fourier_expression(Ff, a, b, q_min,q_max, q_vmin,q_vmax )
        # 再附加E(q)>=0的避碰条件，1e30近似表示没有有效上界
        g = cs.vertcat(g, *pfun_list)
        lbg = cs.vertcat(lbg, *([0.0]*len(pfun_list)))
        ubg = cs.vertcat(ubg, *([1e30]*len(pfun_list)))
        # 以 ||A||_F + ||A^-1||_F 作为条件数相关的平滑优化目标。
        # f1 = cs.simplify(1.0*cs.norm_fro(A) * cs.norm_fro(A_inv))
        f = cs.simplify(1.0*cs.norm_fro(A) + cs.norm_fro(A_inv))
        # IPOPT 接收扁平决策向量；求解后再恢复为两个 Rank×7 系数矩阵。
        x = cs.reshape(cs.vertcat(a,b),(1, 2*Rank*7))
        fc = optas.Function('fc',[a,b],[A])
        # limited-memory 避免显式构造精确 Hessian；MUMPS 解线性子问题。
        opts = {
            'ipopt': {
                'max_iter': 1000,  
                'tol': 1e-8, 
                'acceptable_tol': 1e-6,  
                'acceptable_iter': 50, 
                'linear_solver': 'mumps',  # or 'ma57', 'ma26'
                'mu_strategy': 'adaptive',  
                'dual_inf_tol': 1e-8,  
                'compl_inf_tol': 1e-8,  
                'bound_relax_factor': 0,  
                'hessian_approximation': 'limited-memory',  # quasi-Newton
            },
            'verbose': False,  
        }
        # 标准 NLP：min_x f(x)，同时满足 lbg <= g(x) <= ubg。
        problem = {'x': x,'f':f, 'g': g}
        S = cs.nlpsol('S', 'ipopt', problem, opts)
        return S,lbg,ubg,fc
    
    # 尝试给定初值，返回最优解
    def find_optimal_point_with_start(self, S,lbg, ubg , Rank=5,x_sample_temp = eps* np.random.random (size= (1,70))):  
        init_x0 = copy.deepcopy(x_sample_temp)
        # sol['x']的排列顺序与构造决策向量x时保持一致
        sol = S(x0 = init_x0,lbg = lbg, ubg = ubg)
        _x0_best = sol['x']
        x_split1,x_split2 = cs.vertsplit(cs.reshape(_x0_best,(2*Rank,7)),Rank)
        return x_split1.full(), x_split2.full() 
    
    # 尝试随机初值，返回最优解
    def find_optimal_point_with_randomstart(self, S,lbg, ubg , Rank=5):
        eps = 0.03
        x_sample_temp = eps* np.random.random (size= (1,70))
        return self.find_optimal_point_with_start(S,lbg, ubg , Rank,x_sample_temp)    

    # 生成随机轨迹，作为优化结果的基线
    def trajectory_with_random(self,Rank=5):
        x_sample_temp = 0.3* np.random.random (size= (1,70))
        x_split1,x_split2 = cs.vertsplit(cs.reshape(x_sample_temp,(2*Rank,7)),Rank)
        return x_split1.full(), x_split2.full() 
    
    def generate_opt_traj(self,Ff, sampling_rate, Rank=5, q_min=-1.0*np.ones(7), q_max =3.0*np.ones(7), q_vmin=-5.0*np.ones(7),q_vmax=5.0*np.ones(7), f_path = None, g_path=None):
        """旧版激励轨迹优化实现；功能已由拆分后的优化接口覆盖。"""
        Pb, Pd, Kd = find_dyn_parm_deps(7, 80, self.Ymat)
        K = Pb.T +Kd @Pd.T
        # sampling_rate = 0.1
        pointsNum = int(sampling_rate/(Ff))
        print("pointsNum",pointsNum)
        # raise ValueError("run to here")
        fourierInstance = FourierSeries(ff = Ff)

        a = cs.SX.sym('a', 5,7)
        b = cs.SX.sym('b', 5,7)
        t = cs.SX.sym('t', 1)

        fourierF = fourierInstance.FourierFunction(t, a, b,'f1')
        fourier = fourierF(a,b,t)
        fourierDot = [optas.jacobian(fourier[i],t) for i in range(len(fourier))]
        fourierDDot = [optas.jacobian(fourierDot[i],t) for i in range(len(fourierDot))]
        print(fourierDot)
        path_pos = os.path.join(get_package_share_directory("med7_dock_description"), "meshes", "EndEffector.STL",)
        mesh = o3d.io.read_triangle_mesh(path_pos)
        convex_hull, _ = mesh.compute_convex_hull()
        hull_points = np.asarray(convex_hull.vertices)
        scores = []
        for component_count in range(1, 11):
            model = GaussianMixture(n_components=component_count, random_state=0)
            model.fit(hull_points)
            scores.append(model.score(hull_points))
        best_component_count = int(np.argmax(scores)) + 1
        model = GaussianMixture(n_components=best_component_count, random_state=0)
        model.fit(hull_points)
        points = model.means_
        # print("points", points)
        # raise Exception("Run to here")
        str_prefix = "lbr_"
        vfs_fun = []

        for point in points:
            for i in range(2,6):
                vfs_fun.append(getConstraintsinJointSpace(self.robot, point_coord=point, base_link="link_"+str(i), base_joint_name="A"+str(i)))

        Y_ = []
        Y_fri = []
        pfun_list = []
        for k in range(pointsNum):
            # print("q_np = {0}".format(q_np))
            # q_np = np.random.uniform(-1.5, 1.5, size=7)
            tc = 1.0/(sampling_rate) * k
            # print("tc = ",tc)
            
            q_list = [optas.substitute(id, t, tc) for id in fourier]#fourier(a,b,tc)
            qd_list = [optas.substitute(id, t, tc) for id in fourierDot] #fourierDot(a,b,tc)
            qdd_list = [optas.substitute(id, t, tc) for id in fourierDDot]#fourierDDot(a,b,tc)
            q = cs.vertcat(*q_list)
            qd = cs.vertcat(*qd_list)
            qdd = cs.vertcat(*qdd_list)

            Y_temp = self.Ymat(q, qd, qdd) @Pb
            #[cs.sign(item) for item in qd_list])
            fri_ = cs.diag(cs.sign(qd))
            fri_ = cs.horzcat(fri_,  cs.diag(qd))
            
            for j in range(len(vfs_fun)):
                pfun_list.append(vfs_fun[j](q))

            Y_.append(Y_temp)
            Y_fri.append(fri_)

        Y_r = optas.vertcat(*Y_)
        Y_fri1 = optas.vertcat(*Y_fri)

        # Y = Y_r
        Y = cs.horzcat(Y_r, Y_fri1)
        # print(Y)
        a_eq1 = [0.0]*7
        a_eq2 = [0.0]*7
        b_eq1 = [0.0]*7
        ab_sq_ineq1 = [0.0]*7
        ab_sq_ineq2 = [0.0]*7
        ab_sq_ineq3 = []

        lbg1 = []
        lbg2 = []
        lbg3 = []
        lbg4 = []
        lbg5 = []
        lbg6 = []

        ubg1 = []
        ubg2 = []
        ubg3 = []
        ubg4 = []
        ubg5 = []
        ubg6 = []
        # ab_sq_ineq4 = []
        for i in range(7):
            for l in range(5):
                # print("iter {0}, {1}".format(i, l))
                a_eq1[i] = a_eq1[i] + a[l,i]/(l+1)
                b_eq1[i] = b_eq1[i] + b[l,i]
                a_eq2[i] = a_eq2[i] + a[l,i]*(l+1)

                wl = ((l+1) * Ff* math.pi* 2.0) 
                ab_sq_ineq1[i] = (ab_sq_ineq1[i] + 1.0/(wl)* cs.sqrt(a[l,i]*a[l,i] + b[l,i]*b[l,i]))

                ab_sq_ineq2[i] = (ab_sq_ineq2[i]+ 
                cs.sqrt(a[l,i]*a[l,i] + b[l,i]*b[l,i]))

                ab_sq_ineq3.append(a[l,i])
                ab_sq_ineq3.append(b[l,i])

                cpr2 = min((l+1)*Ff/5.0*2.0*math.pi*q_max[i],q_vmax[i])
                cpr = max((l+1)*Ff/5.0*2.0*math.pi*q_min[i],q_vmin[i])
                lbg6.append(cpr)
                lbg6.append(cpr)

                ubg6.append(cpr2)
                ubg6.append(cpr2)

            lbg1.append(0.0)
            lbg2.append(0.0)
            lbg3.append(0.0)
            lbg4.append(0.0)
            lbg5.append(0.0)

            ubg1.append(0.0)
            ubg2.append(0.0)
            ubg3.append(0.0)
            ubg4.append(q_max[i])
            ubg5.append(q_vmax[i])

        g = cs.vertcat(*(a_eq1+  a_eq2+  b_eq1+  ab_sq_ineq1+ ab_sq_ineq2 + ab_sq_ineq3 +pfun_list))
        lbg = cs.vertcat(*(lbg1,lbg2,lbg3,lbg4,lbg5,lbg6, [0.0]*len(pfun_list)))
        ubg = cs.vertcat(*(ubg1,ubg2,ubg3,ubg4,ubg5,ubg6, [1e10]*len(pfun_list)))

        A = Y.T @ Y
        # print("Y = {0}".format(Y.shape))
        A_inv = cs.inv(A)

        f = cs.simplify(1.0*cs.norm_fro(A) + cs.norm_fro(A_inv))
        x = cs.reshape(cs.vertcat(a,b),(1, 70))
        # fout = objective(a,b)

        fc = optas.Function('fc',[a,b],[A])
        f_fun = optas.Function('ff',[a,b],[f])
        # _f_fun = optas.Function('f_ffc',[a,b],[_f])
        g_fun = optas.Function('gf',[a,b],[g])

        G_max = 3# 
        values_f_min = 10e10
        eps = 2.0

        init_x0_best = eps* np.random.random (size= (1,70))
        reject_sample = 100

        problem = {'x': x,'f':f, 'g': g}
        S = cs.nlpsol('S', 'ipopt', problem,
                      {'ipopt':{'max_iter':50 }, 
                       'verbose':False,
                       "ipopt.hessian_approximation":"limited-memory"
                       })

        for iter in range(G_max):
            for num in range(reject_sample):
                x_sample_temp = eps* np.random.random (size= (1,70))
                init_x0 = copy.deepcopy(x_sample_temp)
                a_init, b_init =  np.split(x_sample_temp.reshape(10,7),2)
                g_data = g_fun(a_init, b_init)

                if(np.all(g_data < ubg) and np.all(g_data > lbg)):
                    print("Find a initial solution here")
                    break
            init_x0 = copy.deepcopy(x_sample_temp)
            a_init, b_init =  np.split(x_sample_temp.reshape(10,7),2)

            sol = S(x0 = init_x0,lbg = lbg, ubg = ubg)
            a_, b_ =  cs.vertsplit(cs.reshape(sol['x'],(10,7)),5)
            values_f = f_fun(a_, b_)
            if values_f_min > values_f:
                print(" find a better value = {0}".format(values_f))
                _x0_best = sol['x']
                values_f_min = values_f
                if (values_f < 1000):
                    break

        x_split1,x_split2 = cs.vertsplit(cs.reshape(_x0_best,(10,7)),5)

        print("sol = {0}".format(_x0_best))
        return x_split1.full(),x_split2.full(),fc

    # 将Fourier系数保存为CSV文件
    def generateToCsv(self, a, b, Ff, sampling_rate, path=None, scale=1.0, bias=None,):
        assert a.shape == b.shape
        if path is None:
            path1 = "/tmp/target_joint_states.csv"
        else:
            path1 = path
        # 统一生成表头和行数据，保存到csv文件
        values_list, keys = self.generateToList(a, b, Ff, sampling_rate, bias=bias)
        with open(path1,"w") as csv_file:
            self.save_(csv_file, keys, values_list)
        return True

    # 根据傅立叶系数、基频、采样率等参数生成傅立叶轨迹列表
    def generateToList(self, a, b, Ff, sampling_rate, scale=1.0, bias=None,):
        assert a.shape == b.shape
        if bias is None:
            bias = np.zeros(7)
        fourierInstance1 = FourierSeries(Rank=a.shape[0], channel=a.shape[1], bias=np.asarray(bias, dtype=float).tolist(), ff=Ff,)
        cs_a = cs.SX.sym('ca', 5, 7)
        cs_b = cs.SX.sym('cb', 5, 7)
        t = cs.SX.sym('tt', 1)
        fourierF = fourierInstance1.FourierFunction(t, cs_a, cs_b,'f2')
        fourier = fourierF(cs_a, cs_b, t)
        # 速度从同一个位置表达式自动微分，保证位置与速度数学一致。
        fourierDot = [optas.jacobian(fourier[i], t) for i in range(len(fourier))]
        _fDot = optas.Function('fund', [cs_a, cs_b, t], fourierDot)
        # 一个周期的点数，终点不写入
        pointsNum = int(sampling_rate/Ff)
        keys = ["关节0位置", "关节1位置", "关节2位置", "关节3位置", "关节4位置", "关节5位置", "关节6位置", 
                "关节0速度", "关节1速度", "关节2速度", "关节3速度", "关节4速度", "关节5速度", "关节6速度"]
        keys = ["时间戳"] + keys
        values_list = []
        for k in range(pointsNum):
            # 采样时间
            tc = 1.0/(sampling_rate) * k
            # 采样位置
            f_temp = fourierInstance1.FourierValue(a, b, scale*tc)
            # 采样速度
            fd_temp=_fDot(np.asarray(a), np.asarray(b), scale*tc)
            # 转化到列表
            q_list = [float(id) for id in f_temp]
            qd_list = [float(id) for id in fd_temp] 
            # 拼接到列表
            values_list.append([tc] + q_list + qd_list)
        return values_list, keys
    
    # 按照keys指定的列顺序写入带表头CSV
    def save_(self, csv_file, keys: List[str], values_list: List[List[float]]) -> None:
        csv_writer = csv.DictWriter(csv_file, fieldnames=keys)
        csv_writer.writeheader()
        for values in values_list:
            csv_writer.writerow({key: value for key, value in zip(keys,values)})

    # 加载已序列化的CasADi目标函数并评价给定系数
    def output_perform_with_full(self, a, b, path_f):
        x = cs.reshape(cs.vertcat(a,b),(1, 70))
        _f = cs.Function.load(path_f)
        f = _f(a,b)
        return f

    # 离线分析保存的目标和约束函数
    def load_analyse_data(self,path_f, path_g):
        a = cs.SX.sym('a', 5, 7)
        b = cs.SX.sym('b', 5, 7)
        x = cs.reshape(cs.vertcat(a,b),(1, 70))
        _f = cs.Function.load(path_f)
        _g = cs.Function.load(path_g)
        f = _f(a,b)
        g = _g(a,b)
        G_max = 10
        eps = 2.0
        init_x0_best = 2.0* np.random.random (size= (1,70))
        def objective_function(pop):
            Alpha1 = 0.75
            Alpha2 = 0.25
            fitness = np.zeros(pop.shape[0])
            for i in range(pop.shape[0]):
                x = copy.deepcopy(pop[i])
                a_init, b_init =  np.split(x.reshape(10,7),2)
                fitness[i] = -Alpha1*_f(a_init, b_init) #- Alpha2*(con_a+con_b)
            return fitness
        
        def selection(pop, fitness, pop_size):
            next_generation = np.zeros((pop_size, pop.shape[1]))
            elite = np.argmax(fitness)
            # print("elite = ",elite)
            next_generation[0] = pop[elite]  # keep the best
            print("pop[elite]  = ",np.max(fitness))
            fitness = np.delete(fitness,elite)
            pop = np.delete(pop,elite,axis=0)
            P = [f/sum(fitness) for f in fitness]  # selection probability
            index = list(range(pop.shape[0]))
            index_selected = np.random.choice(index, size=pop_size-1, replace=False, p=P)
            s = 0
            for j in range(pop_size-1):
                next_generation[j+1] = pop[index_selected[s]]
                s +=1
            return next_generation
        
        def crossover(pop, crossover_rate):
            offspring = np.zeros((crossover_rate, pop.shape[1]))
            for i in range(int(crossover_rate/2)):
                r1=random.randint(0, pop.shape[0]-1)
                r2 = random.randint(0, pop.shape[0]-1)
                while r1 == r2:
                    r1 = random.randint(0, pop.shape[0]-1)
                    r2 = random.randint(0, pop.shape[0]-1)
                cutting_point = random.randint(1, pop.shape[1] - 1)
                offspring[2*i, 0:cutting_point] = pop[r1, 0:cutting_point]
                offspring[2*i, cutting_point:] = pop[r2, cutting_point:]
                offspring[2*i+1, 0:cutting_point] = pop[r2, 0:cutting_point]
                offspring[2*i+1, cutting_point:] = pop[r1, cutting_point:]
            return offspring
        
        def mutation(pop, mutation_rate):
            offspring = np.zeros((mutation_rate, pop.shape[1]))
            for i in range(int(mutation_rate/2)):
                r1=random.randint(0, pop.shape[0]-1)
                r2 = random.randint(0, pop.shape[0]-1)
                while r1 == r2:
                    r1 = random.randint(0, pop.shape[0]-1)
                    r2 = random.randint(0, pop.shape[0]-1)
                cutting_point = random.randint(0, pop.shape[1]-1)
                offspring[2*i] = pop[r1]
                offspring[2*i,cutting_point] = pop[r2,cutting_point]
                offspring[2*i+1] = pop[r2]
                offspring[2*i+1, cutting_point] = pop[r1, cutting_point]
            return offspring
        
        def local_search(pop, n_sol, step_size):
            # number of offspring chromosomes generated from the local search
            offspring = np.zeros((n_sol, pop.shape[1]))
            for i in range(n_sol):
                r1 = np.random.randint(0, pop.shape[0])
                chromosome = pop[r1, :]
                r2 = np.random.randint(0, pop.shape[1])
                chromosome[r2] += np.random.uniform(-step_size, step_size)
                if chromosome[r2] < eps:
                    chromosome[r2] = eps
                if chromosome[r2] > -eps:
                    chromosome[r2] = -eps
                offspring[i,:] = chromosome
            return offspring
        
        rate_crossover = 20         # number of chromosomes that we apply crossower to
        rate_mutation = 20          # number of chromosomes that we apply mutation to
        rate_local_search = 10      # number of chromosomes that we apply local_search to
        step_size = 0.02
        pop_size = 100
        pop = 2.0* np.random.random (size= (pop_size,70))
        
        for iter in range(G_max):
            offspring_from_crossover = crossover(pop, rate_crossover)
            offspring_from_mutation = mutation(pop, rate_mutation)
            offspring_from_local_search = local_search(pop, rate_local_search, step_size)
            
            # we append childrens Q (cross-overs, mutations, local search) to paraents P
            # having parents in the mix, i.e. allowing for parents to progress to next iteration - Elitism
            pop = np.append(pop, offspring_from_crossover, axis=0)
            pop = np.append(pop, offspring_from_mutation, axis=0)
            pop = np.append(pop, offspring_from_local_search, axis=0)
            # print(pop.shape)
            fitness_values = objective_function(pop)
            pop = selection(pop, fitness_values, pop_size)  # we arbitrary set desired pereto front size = pop_size
            print('iteration: {0}  p: [{1}]'.format(iter, pop[0]))

        init_x0_best = pop[0]
        x_temp = copy.deepcopy(init_x0_best)
        a_init, b_init =  np.split(x_temp.reshape(10,7),2)
        output = _f(a_init, b_init)
        print("x_temp=", x_temp)
        raise ValueError("Run to here {0}".format(output))

class TrajGenerationUsrPath(TrajGeneration):
    def __init__(self, path=None, node_name = "para_estimatior", dt_ = 5.0, N_ = 100, gravity_vector=[0, 0, -9.81], ee_link = "link7", ) -> None:
        Node.__init__(self, node_name = node_name)
        if(path is None):
            raise ValueError("This Class need a pathdefine")
        _path = path
        gv = gravity_vector
        self.initial_model_params(_path, gv, ee_link)

def main(args=None):
    rclpy.init(args=args)
    # 根据包名获取share目录下的完整路径
    model_path = os.path.join(get_package_share_directory("xarm_description"), "urdf", "xarm7_description.urdf",)
    # model_path = os.path.join(get_package_share_directory("nero_description"), "urdf", "nero_description.urdf",)
    print("urdf路径 = ", model_path)
    paraEstimator = TrajGenerationUsrPath(path = model_path, gravity_vector = [0, 0, -9.81], ee_link = "link_eef")
    # paraEstimator = TrajGenerationUsrPath(path = model_path, gravity_vector = [0, 0, -9.81], ee_link = "link7")
    # 基频0.1Hz，优化时降采样，导出时使用100Hz
    Ff = 0.1
    sampling_rate = 100.0
    # sampling_rate = 20.0
    sampling_rate = 5.0  
    #   bias = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bias = [0.0, 0.0, 0.0, 2.5, 0.0, 0.7, 0.0]  # xarm7机械臂测试零偏
    a, b, fc = paraEstimator.generate_opt_traj_Link(Ff = Ff,                                                     # 每秒运行周期数
                                                  sampling_rate = sampling_rate,                                 # 每秒采样数
                                                  bias = bias,                 
                                                #   q_min = [-6.28, -2.06, -6.28, -0.19, -6.28, -1.69, -6.28],     # xarm7机械臂真实限位
                                                #   q_max = [ 6.28,  2.06,  6.28,  3.93,  6.28,  3.14,  6.28])     # xarm7机械臂真实限位
                                                  q_min = [-6.28, -2.06, -6.28,  1.00, -6.28, -1.69, -6.28],     # xarm7机械臂测试限位
                                                  q_max = [ 6.28,  2.06,  6.28,  3.93,  6.28,  3.14,  6.28])     # xarm7机械臂测试限位
                                                #   q_min = [-2.7, -1.74, -2.75, -1.01, -2.75, -0.73, -1.571],   # nero机械臂真实限位
                                                #   q_max = [ 2.7,  1.74,  2.75,  1.01,  2.75,  0.95,  1.571])   # nero机械臂真实限位
                                                #   q_min = [-2.7, -2.0, -2.75, -1.6, -2.75, -1.8, -1.7],        # nero机械臂测试限位
                                                #   q_max = [ 2.7,  2.0,  2.75,  1.6,  2.75,  1.8,  1.7])        # nero机械臂测试限位
    print("a = {0} \n b = {1}".format(a, b))
    # 保存轨迹时采样率，根据机械臂控制频率决定
    gen_traj_sampling_rate = 100
    ret = paraEstimator.generateToCsv(a, b, Ff = Ff, sampling_rate = gen_traj_sampling_rate, bias=bias,)
    if ret:
        print("Done! 轨迹已生成（当前碰撞约束关闭）")
        eigenvalues = np.linalg.eigvalsh(np.asarray(fc(a, b).full(), dtype=float))
        print("fc 的正则化特征值 = ", eigenvalues)
        print("a = {0} \n b = {1}".format(a, b))
        conditional_num = np.sqrt(eigenvalues[-1] / eigenvalues[0])
        print("条件数为 = ", conditional_num)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
