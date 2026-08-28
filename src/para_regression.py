#!/usr/bin/python3
import optas
import casadi as cs
from typing import List

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
from dynamic_model import TD_2order, TD_list_filter,find_dyn_parm_deps, RNEA_function,DynamicLinearlization,getJointParametersfromURDF
from scipy import signal
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
Order = [0,1,2,3,4,5,6]


# 从关节位置计算速度和加速度；样本足够时使用 Savitzky-Golay 滤波。
def differentiate_positions(positions, dt, *, window_length=11, polyorder=3):
    q = np.asarray(positions, dtype=float)
    if q.ndim != 2 or q.shape[0] < 2:
        raise ValueError("positions must have shape (samples, joints), samples >= 2")
    if not np.isscalar(dt) or float(dt) <= 0:
        raise ValueError("dt must be a positive scalar")
    n = q.shape[0]
    window = min(int(window_length), n if n % 2 else n - 1)
    if window >= polyorder + 2 and window >= 5:
        qd = signal.savgol_filter(q, window, polyorder, deriv=1, delta=dt,
                                  axis=0, mode="interp")
        qdd = signal.savgol_filter(q, window, polyorder, deriv=2, delta=dt,
                                   axis=0, mode="interp")
    else:
        edge_order = 2 if n >= 3 else 1
        qd = np.gradient(q, dt, axis=0, edge_order=edge_order)
        qdd = np.gradient(qd, dt, axis=0, edge_order=edge_order)
    return qd, qdd


# 对回归矩阵按列缩放后求最小二乘解，避免使用数值稳定性较差的正规方程。
def scaled_least_squares(regressor, target, *, weights=None, rcond=None):
    H = np.asarray(regressor, dtype=float)
    y = np.asarray(target, dtype=float)
    if H.ndim != 2 or y.shape[0] != H.shape[0]:
        raise ValueError("regressor and target sample counts must match")
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != H.shape[0] or np.any(w <= 0):
            raise ValueError("weights must be positive and match the row count")
        root_w = np.sqrt(w)
        H = H * root_w[:, None]
        y = y * root_w.reshape((-1,) + (1,) * (y.ndim - 1))
    scale = np.linalg.norm(H, axis=0)
    scale[scale == 0] = 1.0
    solution, residuals, rank, singular_values = np.linalg.lstsq(
        H / scale, y, rcond=rcond
    )
    solution = solution / scale.reshape((-1,) + (1,) * (solution.ndim - 1))
    return solution, residuals, rank, singular_values

# 执行加权最小二乘法进行参数估计，H: 回归矩阵，大小为 (m, n)  i: 电流数据，大小为 (m, 1)
def weighted_least_squares(H, i, max_iterations=100, tolerance=1e-6):
    return scaled_least_squares(H, i)[0]

# 选择误差最小的n_samples个样本，A: 样本矩阵，M_fri: 摩擦矩阵，b: 样本对应的标签或目标值，n_samples: 选择的重要样本数量
def select_important_samples(A, M_fri, b, preds, n_samples):
    # 将 CasADi DM 转换为 NumPy 数组
    A_np = A.full()
    M_fri_np = M_fri
    b_np = b
    # 把预测值转换为一维数组
    predictions = preds.full().flatten()
    print("predictions = ", predictions.shape)
    # 通过预测值计算误差
    errors = np.abs(predictions - b_np)
    print("errors = ", errors.shape)
    print("b_np = ", b_np.shape)
    # 选择误差最大的n_samples个样本的索引
    important_indices = np.argsort(errors)[-n_samples:]
    A_important = A_np[important_indices, :]
    M_fri_imp = M_fri_np[important_indices, :]
    b_important = b_np[important_indices]
    print("M_fri_imp = ",M_fri_imp.shape)
    print("A_np = ",A_np.shape)
    print("M_fri_np = ",M_fri_np.shape)
    print("A_important = ",A_important.shape)
    print("important_indices = ",important_indices.shape)
    return A_important, M_fri_imp, b_important

# 动力学参数估计器
class Estimator():
    def __init__(self, path, ee_link="link_eef", node_name = "para_estimatior", dt_ = 5.0, N_ = 100, gravity_vec = [0.0, 0.0, -9.81]) -> None:
        self.dt_ = dt_
        self.N = N_
        # 获取机器人模型
        self.robot = optas.RobotModel(urdf_filename=path, time_derivs=[1])
        # 获取机器人关节参数
        Nb, xyzs, rpys, axes = getJointParametersfromURDF(self.robot, ee_link=ee_link)
        # 计算机器人动力学模型
        self.dynamics_ = RNEA_function(Nb, 1, rpys, xyzs, axes, gravity_para = cs.DM(gravity_vec))
        # 通过动态线性化方法获取动力学回归矩阵和参数向量
        self.Ymat, self.PIvector = DynamicLinearlization(self.dynamics_, Nb)
        # 读取urdf
        urdf_string_ = pathlib.Path(path).read_text(encoding="utf-8")
        robot = urdf.URDF.from_xml_string(urdf_string_)
        # 获取每个连杆的质量
        masses = [link.inertial.mass for link in robot.links if link.inertial is not None]
        # 为nero补充一个接近零质量的虚拟末端刚体(nero没有末端固定刚体)
        self.masses_np = np.append(np.asarray(masses[1:], dtype=float), 1e-6)
        # print("masses = {0}".format(self.masses_np))
        # 获取每个连杆的质心向量
        massesCenter = [link.inertial.origin.xyz for link in robot.links if link.inertial is not None]
        self.massesCenter_np = np.column_stack((np.asarray(massesCenter[1:], dtype=float).T, np.zeros(3)))
        # 获取每个连杆的惯性参数
        Inertia = [link.inertial.inertia.to_matrix() for link in robot.links if link.inertial is not None]
        self.Inertia_np = np.hstack((*Inertia[1:], np.eye(3, dtype=float) * 1e-5))
        
    @ staticmethod
    def readCsvToList(path):
        l = []
        with open(path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                joint_names = [x.strip() for x in list(row.keys())]
                l.append([float(x) for x in row.values()])
        return l    
    
    # 从CSV文件读取关节位置和力矩，然后计算关节速度
    def ExtractFromMeasurmentCsv(self, path_pos):
        dt = 0.01
        pos_l = []
        tau_ext_l = []
        with open(path_pos) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # print("111 = {0}".format(row.values()))
                pl = list(row.values())[0:7]
                tl = list(row.values())[7:14]
                joint_names = [x.strip() for x in list(row.keys())]
                pos_l.append([float(x) for x in pl])
                tau_ext_l.append([float(x) for x in tl])
        vel_l, _ = differentiate_positions(pos_l, dt)
        return pos_l, vel_l.tolist(), tau_ext_l
    
    # 从测量数据中提取关节位置和关节力矩，并通过位置差分计算关节速度
    def ExtractFromMeasurmentList(self, pos_list):
        dt = 0.01
        pos_l = []
        tau_ext_l = []
        for row in pos_list:
            # print("111 = {0}".format(row.values()))
            pl = row[0:7]
            tl = row[7:14]
            pos_l.append([float(x) for x in pl])
            tau_ext_l.append([float(x) for x in tl])
        vel_l, _ = differentiate_positions(pos_l, dt)
        return pos_l, vel_l.tolist(), tau_ext_l

    # 从测量数据中提取七个关节的位置和外力矩，并把所有关节速度设置为0
    def ExtractFromMeasurmentListZeroVel(self, pos_list):
        dt = 0.01
        pos_l = []
        tau_ext_l = []
        for row in pos_list:
            # print("111 = {0}".format(row.values()))
            pl = row[0:7]
            tl = row[7:14]
            pos_l.append([float(x) for x in pl])
            tau_ext_l.append([float(x) for x in tl])
        vel_l =[]
        for id in range(len(pos_l)):
            vel_l.append([0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0])
        return pos_l, vel_l, tau_ext_l    
       
    # 把数据以字典行的形式写入CSV文件
    def save_(self, csv_file, keys: List[str], values_list: List[List[float]]) -> None:
        csv_writer = csv.DictWriter(csv_file, fieldnames=keys)
        csv_writer.writeheader()
        for values in values_list:
            csv_writer.writerow({key: value for key, value in zip(keys, values)})
    
    def timer_cb_regressor_physical_con_impt_samp(self, positions, velocities, efforts):
        Pb, Pd, Kd = find_dyn_parm_deps(7, 80, self.Ymat)
        K = Pb.T +Kd @Pd.T
        taus = []
        Y_ = []
        Y_fri = []
        if len(positions) < 2:
            raise ValueError("at least two samples are required")
        for k in range(1, len(positions)):
            q_np = [positions[k][i] for i in Order]
            qd_np = [velocities[k][i] for i in Order]
            tau_ext = [efforts[k][i] for i in Order]
            qdlast_np = [velocities[k-1][i] for i in Order]
            qdd_np = (np.array(qd_np)-np.array(qdlast_np))/0.01
            qdd_np_list = qdd_np.tolist()
            Y_temp = self.Ymat(q_np, qd_np, qdd_np_list) @Pb 
            fri_ = np.diag([float(np.sign(item)) for item in qd_np])
            fri_ = np.hstack((fri_,  np.diag(qd_np)))
            Y_.append(Y_temp)
            taus.append(tau_ext)
            Y_fri.append(np.asarray(fri_))
        Y_r = optas.vertcat(*Y_)
        taus1 = np.hstack(taus)
        Y_fri1 = np.vstack(Y_fri)
        pa_size = Y_r.shape[1]
        taus1 = taus1.T
        Y = cs.DM(np.hstack((Y_r, Y_fri1)))
        _w1, _h1 =self.massesCenter_np.shape
        _w2, _h2 =self.Inertia_np.shape
        _w0 = len(self.masses_np)
        l1 = _w0 + _w1*_h1
        l2 = _w0 + _h1*_w1 + _w2 * _h2
        # with friction
        l = l2+ len(qd_np)*2
        _estimate = cs.SX.sym('para', l)
        estimate_cs = K @ self.PIvector(_estimate[0:_w0], _estimate[_w0:l1].reshape((_w1,_h1)), _estimate[l1:l2].reshape((_w2,_h2)))
        e_cs_fun = cs.Function('ecs',[_estimate], [estimate_cs])
        obj = cs.sumsqr(taus1 - Y_r @ estimate_cs -Y_fri1 @ _estimate[-len(qd_np)*2:])+ 2.0 * cs.norm_2(_estimate[:_w0])+ 5.0 * cs.norm_2(_estimate[_w0:l1]) + 5.0 * cs.norm_2(_estimate[l1:l2])
        mass_norminal = self.masses_np
        mass_center_norminal = self.massesCenter_np.reshape(-1,_w1*_h1).flatten()
        intertia_norminal = self.Inertia_np.reshape(-1,_w2*_h2).flatten()
        Inertia = _estimate[l1:l2].reshape((_w2,_h2))
        print("_w2, _h2 = {0}, {1}".format(_w2, _h2))
        list_of_intertia_norminal = [Inertia[:, i:i+3] for i in range(0, Inertia.shape[1], 3)]
        print("list_of_intertia_norminal = ",list_of_intertia_norminal)
        ineq_constr = []
        ineq_constr += [_estimate[i]> 0.0 for i in range(_w0)]
        for I in list_of_intertia_norminal:
            Ii = cs.eig_symbolic(I)
            ineq_constr += [Ii[id]>0.0 for id in range(3)]
        ineq_constr += [I[0,0] <=I[1,1] +I[2,2] for I in list_of_intertia_norminal]
        ineq_constr += [I[1,1] <=I[0,0] +I[2,2] for I in list_of_intertia_norminal]
        ineq_constr += [I[2,2] <=I[1,1] +I[0,0] for I in list_of_intertia_norminal]
        ineq_constr += [100.0*cs.mmin(cs.vertcat(I[1,1], I[0,0], I[2,2]))  >=cs.mmax(cs.vertcat(I[1,1], I[0,0], I[2,2])) for I in list_of_intertia_norminal]
        ineq_constr += [3.0 * list_of_intertia_norminal[j][2,2]<= cs.mmin(cs.vertcat(list_of_intertia_norminal[j][0,0], list_of_intertia_norminal[j][1,1])) for j in [0, 2, 4]]
        ineq_constr += [3.0 * list_of_intertia_norminal[k][1,1]<= cs.mmin(cs.vertcat(list_of_intertia_norminal[k][0,0], list_of_intertia_norminal[k][2,2])) for k in [1, 3]]
        ineq_constr += [1e-4<= I[0,0] for I in list_of_intertia_norminal]
        ineq_constr += [1e-4<= I[1,1] for I in list_of_intertia_norminal]
        ineq_constr += [1e-4<= I[2,2] for I in list_of_intertia_norminal]
        ineq_constr += [cs.mmax(cs.vertcat(cs.norm_2(I[1,0]), cs.norm_2(I[0,2]), cs.norm_2(I[1,2])))<= 0.1*cs.norm_2(cs.mmin(cs.vertcat(I[1,1], I[0,0], I[2,2]))) for I in list_of_intertia_norminal]
        problem = {'x': _estimate, 'f': obj, 'g': cs.vertcat(*ineq_constr)}
        opts = {
            'ipopt': {
                'max_iter': 1000,
                'tol': 1e-8,
                'acceptable_tol': 1e-6,
                'acceptable_iter': 10,
                'linear_solver': 'mumps',  # 或其他高效线性求解器，如 'ma57', 'ma86','mumps'
                'hessian_approximation': 'limited-memory',
            },
            'verbose': False,
        }
        # 创建求解器
        solver = cs.nlpsol('S', 'ipopt', problem, opts)
        print("solver = {0}".format(solver))
        gt_x0 = mass_norminal.tolist() + mass_center_norminal.tolist() + intertia_norminal.tolist() + [0.1]*len(qd_np) + [0.5]*len(qd_np)
        import random
        init_x0 = (mass_norminal*np.random.uniform(1.5, 3.5, size=mass_norminal.shape)).tolist() 
        + (mass_center_norminal*np.random.uniform(0.0, 0.2, size=mass_center_norminal.shape)).tolist() 
        + (intertia_norminal*np.random.uniform(0.0, 0.1, size=intertia_norminal.shape)).tolist() 
        + [random.random()*0.05 for _ in range(len(qd_np))]
        + [random.random()*0.2 for _ in range(len(qd_np))]
        sol = solver(x0 = init_x0)
        preds = taus1 - Y_r @ e_cs_fun(sol['x']) -Y_fri1 @ sol['x'][-len(qd_np)*2:]
        Y_r1, Y_fri2, taus2 = select_important_samples(Y_r, Y_fri1, taus1, preds, 140)
        print("Y_fri2 = ", Y_fri2.shape)
        obj2 = cs.sumsqr(taus2 - Y_r1 @ estimate_cs -Y_fri2 @ _estimate[-len(qd_np)*2:])+ 2.0 * cs.norm_2(_estimate[:_w0]) + 5.0 * cs.norm_2(_estimate[_w0:l1]) + 5.0 * cs.norm_2(_estimate[l1:l2])
        problem2 = {'x': _estimate, 'f': obj2, 'g': cs.vertcat(*ineq_constr)}
        solver2 = cs.nlpsol('S', 'ipopt', problem2, opts)
        sol2 = solver2(x0 = sol['x'])
        return sol2['x'], np.array(gt_x0)
    
    # 根据关节位置、速度和测量力矩，构造参数辨识所需的三个量
    def get_Yb_matrix(self, positions, velocities, efforts, Pb):
        taus = []
        Y_ = []
        Y_fri = []
        if len(positions) < 2:
            raise ValueError("at least two samples are required")
        # 从第二个样本开始遍历
        for k in range(1, len(positions)):
            q_np = [positions[k][i] for i in Order]
            qd_np = [velocities[k][i] for i in Order]
            tau_ext = [efforts[k][i] for i in Order]
            qdlast_np = [velocities[k-1][i] for i in Order]
            # 求加速度
            qdd_np = (np.array(qd_np) - np.array(qdlast_np))/0.01
            qdd_np_list = qdd_np.tolist()
            # 求最小参数集对应的回归矩阵行
            Y_temp = self.Ymat(q_np, qd_np, qdd_np_list) @ Pb 
            # 构造摩擦回归矩阵
            fri_ = np.diag([float(np.sign(item)) for item in qd_np]) # 速度方向
            fri_ = np.hstack((fri_,  np.diag(qd_np)))                # 把速度方向和速度大小水平堆叠
            Y_.append(Y_temp)
            taus.append(tau_ext)
            Y_fri.append(np.asarray(fri_))
        Y_r = optas.vertcat(*Y_)
        taus1 = np.hstack(taus).T
        Y_fri1 = np.vstack(Y_fri)
        return Y_r, taus1, Y_fri1
    
    # 对惯性参数构建不等式物理约束
    @staticmethod
    def build_ineq_physical_con(_estimate, _w0, _w1, _h1, _w2, _h2):
        # 质量参数和质心参数数量和
        l1 = _w0 + _w1 *_h1
        # 所有惯性参数数量和
        l2 = l1 + _w2 * _h2
        # 得到惯量矩阵
        Inertia = _estimate[l1:l2].reshape((_w2, _h2))
        # 每三列切分为一个连杆的3X3惯量矩阵
        list_of_intertia_norminal = [Inertia[:, i:i+3] for i in range(0, Inertia.shape[1], 3)]
        constraints, lower, upper = [], [], []
        # 辅助函数用于构造约束列表
        def bounded(expression, lb = -np.inf, ub = np.inf):
            constraints.append(expression)
            lower.append(lb)
            upper.append(ub)
        # 对质量构建约束，大于0
        for i in range(_w0):
            bounded(_estimate[i], 1e-6, np.inf)
        # 对转动惯量构建约束
        for I in list_of_intertia_norminal:
            # 必须对称
            bounded(I[0, 1] - I[1, 0], 0.0, 0.0)
            bounded(I[0, 2] - I[2, 0], 0.0, 0.0)
            bounded(I[1, 2] - I[2, 1], 0.0, 0.0)
            # 惯量矩阵必须正定
            # 一阶主子式大于0
            bounded(I[0, 0], 1e-8, np.inf)
            # 二阶顺序主子式大于0
            bounded(I[0, 0] * I[1, 1] - I[0, 1] ** 2, 1e-12, np.inf)
            # 三阶顺序主子式大于0
            bounded(cs.det(I), 1e-15, np.inf)
            # 任意一个方向的惯量，都不能大于另外两个方向惯量之和
            bounded(I[1, 1] + I[2, 2] - I[0, 0], 0.0, np.inf)
            bounded(I[0, 0] + I[2, 2] - I[1, 1], 0.0, np.inf)
            bounded(I[0, 0] + I[1, 1] - I[2, 2], 0.0, np.inf)
        # 库伦摩擦和黏滞摩擦系数必须大于0
        for i in range(l2, _estimate.numel()):
            bounded(_estimate[i], 0.0, np.inf)
        return constraints, lower, upper
    
    # 拼接仿真中的名义物理参数和预设摩擦参数，便于与辨识出的参数做对比
    @staticmethod
    def get_gt_params_sim(mass_norminal, mass_center_norminal, intertia_norminal, nj, fri_p1 = 0.1, fri_p2 = 0.5):
        gt_x0 = mass_norminal.tolist() + mass_center_norminal.tolist() + intertia_norminal.tolist() + [fri_p1]*nj + [fri_p2]*nj
        return gt_x0
    
    # 从机器人模型中读取惯性参数，再补上默认摩擦参数，拼接成完整的仿真真值参数向量
    def get_gt_params_simO(self):
        nj = self.robot.ndof
        mass_norminal = self.masses_np
        _w1, _h1 = self.massesCenter_np.shape
        _w2, _h2 = self.Inertia_np.shape
        mass_center_norminal = self.massesCenter_np.reshape(-1, _w1*_h1).flatten()
        intertia_norminal = self.Inertia_np.reshape(-1, _w2*_h2).flatten()
        gt_x0 = Estimator.get_gt_params_sim(mass_norminal, mass_center_norminal, intertia_norminal, nj)
        return gt_x0

    # 通过优化求解器进行动力学参数辨识，考虑物理约束
    def timer_cb_regressor_physical_con(self, positions, velocities, efforts):
        nj = len(positions[0])
        # 获取动力学参数独立矩阵
        Pb, Pd, Kd = find_dyn_parm_deps(7, 80, self.Ymat)
        # 由Y*Pd = Y*Pb*Kd推导得到，K左乘完整惯性参数可以得到最小参数集
        K = Pb.T + Kd @ Pd.T
        # 根据位置、速度、关节力矩，计算回归矩阵、关节力矩向量、摩擦参数回归矩阵
        Y_r, taus1, Y_fri1 = self.get_Yb_matrix(positions, velocities, efforts, Pb)
        print("self.masses_np = ", self.masses_np)
        _w1, _h1 = self.massesCenter_np.shape
        _w2, _h2 = self.Inertia_np.shape
        # 质量参数数量
        _w0 = len(self.masses_np)
        # 质量参数和质心参数数量和
        l1 = _w0 + _w1 * _h1
        # 所有惯性参数数量和
        l2 = l1 + _w2 * _h2
        # 待辨识参数长度，带摩擦参数
        l = l2 + nj*2
        # 生成待辨识参数符号向量
        _estimate = cs.SX.sym('para', l)
        # 把待优化的物理参数转换成最小动力学参数
        estimate_cs = K @ self.PIvector(_estimate[0:_w0], _estimate[_w0:l1].reshape((_w1, _h1)), _estimate[l1:l2].reshape((_w2, _h2)))
        # 定义参数辨识的目标函数，包括力矩拟合误差平方和，物理参数正则化两部分，
        obj = (cs.sumsqr(taus1 - Y_r @ estimate_cs - Y_fri1 @ _estimate[-nj*2:])                                    # 实际力矩、减去估计力矩、减去摩擦力
        + 10.0*cs.norm_2(_estimate[:_w0]) + 100.0*cs.norm_2(_estimate[_w0:l1]) + 100.0*cs.norm_2(_estimate[l1:l2])) # 参数正则化部分
        # 为待辨识参数生成约束表达式、约束下界、约束上界
        ineq_constr, constraint_lb, constraint_ub = Estimator.build_ineq_physical_con(_estimate, _w0, _w1, _h1, _w2, _h2)
        # 优化问题: x 决策变量，f 要最小化的标量目标函数，g 约束表达式
        problem = {'x': _estimate, 'f': obj, 'g': cs.vertcat(*ineq_constr)}
        opts = {
            'ipopt': {
                'max_iter': 1000,                                # 最大迭代次数
                'tol': 1e-6,                                     # 总体收敛容差
                'acceptable_tol': 1e-4,                          # 可接受解容差
                'acceptable_iter': 10,                           # 迭代满当前次数后，如果满足可接受解容差，结束迭代
                'constr_viol_tol': 1e-5,                         # 约束违反容差
                'linear_solver': 'mumps',                        # 求解器
                'mu_strategy': 'adaptive',                       # 策略
                'hessian_approximation': 'limited-memory',       # 使用精确的Hessian，不使用近似
                "nlp_scaling_method": "gradient-based",
            },
            'verbose': False,                                    # 如果需要调试信息，可以设置为True
        }
        # 创建求解器: S 名称，ipopt 求解后端，problem 优化问题，opts 配置
        solver = cs.nlpsol('S', 'ipopt', problem, opts)
        print("solver = {0}".format(solver))
        # 整理名义动力学参数
        mass_norminal = self.masses_np
        mass_center_norminal = self.massesCenter_np.reshape(-1, _w1*_h1).flatten()
        intertia_norminal = self.Inertia_np.reshape(-1, _w2*_h2).flatten()
        # 生成用于仿真对比的真实参数向量gt_x0
        gt_x0 = Estimator.get_gt_params_sim(mass_norminal, mass_center_norminal, intertia_norminal, nj)
        import random
        # 初始化质量、质心、惯性参数、摩擦
        init_x0 = (
            (mass_norminal * np.random.uniform(0.0, 2.0, size=mass_norminal.shape)).tolist()
            + (mass_center_norminal * np.random.uniform(0.0, 2.0, size=mass_center_norminal.shape)).tolist()
            + (intertia_norminal * np.random.uniform(0.0, 2.0, size=intertia_norminal.shape)).tolist()
            + [random.random() * 1.0 for _ in range(nj)]
            + [random.random() * 1.0 for _ in range(nj)]
        )
        # 求解优化问题，要满足 lbg <= g(x) <= ubg 和 lbx <= x <= ubx
        sol = solver(x0 = init_x0, lbg = constraint_lb, ubg = constraint_ub)
        stats = solver.stats()
        # if not stats.get('success', False):
        #     raise RuntimeError("带物理约束的优化失败: " + str(stats.get('return_status', 'unknown status')))
        return sol['x'], np.array(gt_x0)
    
    # 使用最小二乘法估计动力学参数
    def timer_cb_regressor(self, positions, velocities, efforts):
        # 获取动力学参数独立性矩阵
        Pb, Pd, Kd = find_dyn_parm_deps(7, 80, self.Ymat)
        K = Pb.T +Kd @ Pd.T
        taus = []
        Y_ = []
        Y_fri = []
        for k in range(1, len(positions)):
            q_np = [positions[k][i] for i in Order]
            qd_np = [velocities[k][i] for i in Order]
            tau_ext = [efforts[k][i] for i in Order]
            qdlast_np = [velocities[k-1][i] for i in Order]
            qdd_np = (np.array(qd_np)-np.array(qdlast_np))/0.01
            qdd_np = qdd_np.tolist()
            Y_temp = self.Ymat(q_np, qd_np, qdd_np) @Pb 
            fri_ = np.diag([float(np.sign(item)) for item in qd_np])
            fri_ = np.hstack((fri_,  np.diag(qd_np)))
            # fri_ = [[np.sign(v), v] for v in qd_np]
            Y_.append(Y_temp)
            taus.append(tau_ext)
            Y_fri.append(np.asarray(fri_))
            # print(qdd_np)
        Y_r = optas.vertcat(*Y_)
        taus1 = np.hstack(taus)
        Y_fri1 = np.vstack(Y_fri)
        print("Y_fri1 = {0}".format(Y_fri1))
        print("Y_fri1 = {0}".format(Y_fri1.shape))
        print("Y_r = {0}".format(Y_r.shape))
        print("Pb = {0}".format(Pb.shape))
        pa_size = Y_r.shape[1]
        taus1 = taus1.T
        # estimate_pam = np.linalg.inv(Y_r.T @ Y_r) @ Y_r.T @ taus1
        Y = cs.DM(np.hstack((Y_r, Y_fri1)))
        estimate_pam = scaled_least_squares(np.asarray(Y), taus1)[0]
        estimate_cs = cs.SX.sym('para', pa_size+14)
        obj = cs.sumsqr(taus1 - Y @ estimate_cs)

        lb = -3.0*np.array([1.0]*(pa_size+14))
        ub = 3.0*np.array([1.0]*(pa_size+14))
        print("self.masses_npv", self.masses_np.shape)
        ref_pam = K @ self.PIvector(self.masses_np,self.massesCenter_np,self.Inertia_np).toarray().flatten()
        print("ref_pam = ",ref_pam.shape)
        print("lb = ",lb.shape)
        
        lb[:pa_size] = -2.0*ref_pam
        ub[:pa_size] = 2.0*ref_pam

        ineq_constr = [estimate_cs[i] >= lb[i] for i in range(pa_size)] + [estimate_cs[i] <= ub[i] for i in range(pa_size)]

        problem = {'x': estimate_cs, 'f': obj, 'g': cs.vertcat(*ineq_constr)}
        solver = cs.nlpsol('S', 'ipopt', problem,{'ipopt':{'max_iter':3000000 }, 'verbose':True})
        print("solver = {0}".format(solver))
        sol = solver()
        print("sol = {0}".format(sol['x']))
        return sol['x'], estimate_pam
    
    # 使用估计的参数计算的关节力矩与实际测量的关节力矩之间的误差
    def testWithEstimatedParaIDyn(self, positions, velocities, para_gt, para)->None:
        # 获取动力学参数独立性矩阵
        Pb, Pd, Kd =find_dyn_parm_deps(7,80,self.Ymat)
        K = Pb.T +Kd @Pd.T
        tau_ests = []
        es = []
        tau_exts = []
        filter_list = [TD_2order(T=0.01) for i in range(7)]
        _w1, _h1 =self.massesCenter_np.shape
        _w2, _h2 =self.Inertia_np.shape
        _w0 = len(self.masses_np)
        l = _w0 + _h1*_w1 + _w2 * _h2
        l1 = _w0 + _w1*_h1
        # 构造最小惯性参数集
        estimate_cs = K @ self.PIvector(para[0:_w0], para[_w0:l1].reshape((_w1,_h1)), para[l1:l].reshape((_w2,_h2)))
        estimate_gt = K @ self.PIvector(para_gt[0:_w0], para_gt[_w0:l1].reshape((_w1,_h1)), para_gt[l1:l].reshape((_w2,_h2)))
        for k in range(1,len(positions),1):
            q_np = [positions[k][i] for i in Order]
            qd_np = [velocities[k][i] for i in Order]
            qdlast_np = [velocities[k-1][i] for i in Order]
            qdd_np = (np.array(qd_np) - np.array(qdlast_np))/0.01
            pa_size = Pb.shape[1]
            # 由模型计算各个关节的力矩
            tau_est_model = (self.Ymat(q_np, qd_np, qdd_np) @ Pb @ estimate_cs 
                             + np.diag(np.sign(qd_np)) @ para[-2*len(qd_np):-len(qd_np)] 
                             + np.diag(qd_np) @ para[-len(qd_np):])
            tau_ext = (self.Ymat(q_np, qd_np, qdd_np) @ Pb @ estimate_gt 
                       + np.diag(np.sign(qd_np)) @ para_gt[-2*len(qd_np):-len(qd_np)] 
                       + np.diag(qd_np) @ para_gt[-len(qd_np):])
            # 计算差值
            e = tau_est_model - tau_ext 
            print("sim_tau = {0}".format(tau_ext))
            print("tau_est_model = {0}".format(tau_est_model))
            print("sim_tau 2 = {0}".format(self.dynamics_(q_np,qd_np, qdd_np, self.masses_np, para_gt[_w0:l1].reshape((_w1,_h1)), para_gt[l1:l].reshape((_w2,_h2)))))
            tau_ests.append(tau_est_model.toarray().flatten().tolist())
            es.append(e.toarray().flatten().tolist())
            tau_exts.append(tau_ext.toarray().flatten().tolist())
        return tau_ests, tau_exts
    
    # 使用估计的参数计算的关节力矩与实际测量的关节力矩之间的误差
    def testWithEstimatedParaCon(self, positions, velocities, efforts, para)->None:
        # 获取动力学参数独立性矩阵
        Pb, Pd, Kd = find_dyn_parm_deps(7, 80, self.Ymat)
        K = Pb.T +Kd @ Pd.T
        tau_ests = []
        es = []
        # 使用二阶低通滤波器对速度进行滤波
        filter_list = [TD_2order(T = 0.01) for i in range(7)]
        _w1, _h1 = self.massesCenter_np.shape
        _w2, _h2 = self.Inertia_np.shape
        _w0 = len(self.masses_np)
        l = _w0 + _h1*_w1 + _w2 * _h2
        l1 = _w0 + _w1*_h1
        # 构造最小惯性参数集
        estimate_cs = K @ self.PIvector(para[0:_w0], para[_w0:l1].reshape((_w1,_h1)), para[l1:l].reshape((_w2,_h2)))
        for k in range(1, len(positions), 1):
            q_np = [positions[k][i] for i in Order]
            qd_np = [velocities[k][i] for i in Order]
            tau_ext = [efforts[k][i] for i in Order]
            qdlast_np = [velocities[k-1][i] for i in Order]
            # 计算角加速度
            qdd_np = (np.array(qd_np) - np.array(qdlast_np))/0.01
            pa_size = Pb.shape[1]
            # 由模型计算各个关节的力矩
            tau_est_model = (self.Ymat(q_np, qd_np, qdd_np) @ Pb @ estimate_cs 
                             + np.diag(np.sign(qd_np)) @ para[-2*len(qd_np):-len(qd_np)] 
                             + np.diag(qd_np) @ para[-len(qd_np):])
            # 计算值与实际值做差
            e= tau_est_model - tau_ext 
            print("sim_tau = {0}".format(tau_ext))
            print("tau_est_model = {0}".format(tau_est_model))
            # print("tau_error = {0}".format(e))
            print("q_np = {0}".format(q_np))
            # 保存到列表
            tau_ests.append(tau_est_model.toarray().flatten().tolist())
            es.append(e.toarray().flatten().tolist())
        return tau_ests, es
    
    # 使用估计的参数计算的关节力矩与实际测量的关节力矩之间的误差
    def testWithEstimatedPara(self, positions, velocities, efforts, para)->None:
        # 获取动力学参数独立性矩阵
        Pb, Pd, Kd =find_dyn_parm_deps(7, 80, self.Ymat)
        K = Pb.T +Kd @Pd.T
        tau_ests = []
        es = []
        # 使用二阶低通滤波器对速度进行滤波
        filter_list = [TD_2order(T=0.01) for i in range(7)]
        for k in range(1,len(positions),1):
            q_np = [positions[k][i] for i in Order]
            qd_np = [velocities[k][i] for i in Order]
            tau_ext = [efforts[k][i] for i in Order]
            qdlast_np = [velocities[k-1][i] for i in Order]
            qdd_np = (np.array(qd_np) - np.array(qdlast_np))/0.01   
            qdd_np = [f(qd_np[id])[1] for id, f in enumerate(filter_list)]
            pa_size = Pb.shape[1]
            tau_est_model = (self.Ymat(q_np,qd_np,qdd_np) @Pb@  para[:pa_size] + np.diag(np.sign(qd_np)) @ para[pa_size:pa_size+7] + np.diag(qd_np) @ para[pa_size+7:])
            e= tau_est_model - tau_ext 
            print("error1 = {0}".format(e))
            print("tau_ext = {0}".format(tau_ext))
            tau_ests.append(tau_est_model.toarray().flatten().tolist())
            es.append(e.toarray().flatten().tolist())
        return tau_ests, es

    # 将估计的参数保存到CSV文件中
    def saveEstimatedPara(self, parac)->None:
        path1 = pathlib.Path(__file__).resolve().parent / "test_data" / "DynamicParameters.csv"
        path1.parent.mkdir(parents=True, exist_ok=True)
        para = np.asarray(parac).reshape(-1)
        body_count = len(self.masses_np)
        centers = para[body_count:body_count * 4].reshape((3, body_count), order="F")
        inertias = para[body_count * 4:body_count * 13].reshape((3, body_count * 3), order="F")
        keys = ["link", "mass", "com_x", "com_y", "com_z", "ixx", "ixy", "ixz", "iyx", "iyy", "iyz", "izx", "izy", "izz"]
        rows = []
        for i in range(7):
            values = [para[i], *centers[:, i], *inertias[:, i * 3:(i + 1) * 3].reshape(-1)]
            rows.append([f"link{i + 1}", *[f"{value:.5f}" for value in values]])
        with path1.open("w", newline="", encoding="utf-8") as csv_file:
            self.save_(csv_file, keys, rows)
            
# 使用butterworth滤波器对轨迹数据进行低通滤波
def traj_filter(states):
    cols = []
    l = len(states[0])
    fs = 100
    cutoff_freq = 2  # 截止频率为10Hz
    b, a = signal.butter(4, cutoff_freq / (fs / 2), 'low')
    filtered_signal = []
    states_filtered = []
    for i in range(l):
        cols.append([float(state[i]) for state in states])
        filtered_signal.append(signal.filtfilt(b, a, cols[i]))
    for j in range(len(filtered_signal[0])):
        states_filtered.append([filtered_signal[i][j] for i in range(l)])
    return states_filtered

# 比较轨迹，绘制估计的外部力和实际的外部力
def compare_traj(states1, states2):
    col1s , col2s = [], []
    l = len(states1[0])
    fig, axs = plt.subplots(7, 1, figsize=(8, 10))
    for i in range(l):
        print("states = {0}".format(states2[i]))
        col1s.append([float(state[i]) for state in states1])
        col2s.append([float(state[i]) for state in states2])
        axs[i].plot(col1s[i])
        axs[i].plot(col2s[i])
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def main(args=None):
    # model_path = os.path.join(get_package_share_directory("nero_description"), "urdf",  "nero_description.urdf",)
    model_path = os.path.join(get_package_share_directory("xarm_description"), "urdf",  "xarm7_description.urdf",)
    # 获取机器人参数估计器
    paraEstimator = Estimator(model_path)
    # 获取数据
    path_pos = pathlib.Path(__file__).resolve().parent / "test_data" / "mujoco_robot_data.csv"
    # 从CSV文件中提取位置、速度和努力数据
    positions, velocities, efforts = paraEstimator.ExtractFromMeasurmentCsv(path_pos)
    velocities = traj_filter(velocities)
    efforts_f = traj_filter(efforts)
    # 进行参数估计
    estimate_pam, ref_pam = paraEstimator.timer_cb_regressor_physical_con(positions, velocities, efforts_f)
    print("estimate_pam = {0}".format(estimate_pam))
    # 进行测试，使用估计的参数进行控制
    tau_exts, es = paraEstimator.testWithEstimatedParaCon(positions, velocities, efforts_f,estimate_pam)
    # 保存估计的参数到CSV文件
    paraEstimator.saveEstimatedPara(estimate_pam)
    # 比较轨迹，绘制估计的外部力和实际的外部力
    compare_traj(tau_exts, efforts_f)

    # path_pos_2 = os.path.join(get_package_share_directory("gravity_compensation"), "test", "measurements_0dgr.csv", )
    # # 从CSV文件中提取位置、速度和努力数据
    # positions_, velocities_, efforts_ = paraEstimator.ExtractFromMeasurmentCsv(path_pos_2)
    # velocities_=traj_filter(velocities_)
    # efforts_f_=traj_filter(efforts_)
    # # 进行测试，使用估计的参数进行控制
    # tau_exts_, es =paraEstimator.testWithEstimatedParaCon(positions_, velocities_, efforts_f_,estimate_pam)
    # # 比较轨迹，绘制估计的外部力和实际的外部力
    # compare_traj(tau_exts_, efforts_f_)

if __name__ == "__main__":
    main()
