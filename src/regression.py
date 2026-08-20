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
from IDmodel import TD_2order, TD_list_filter,find_dyn_parm_deps, RNEA_function,DynamicLinearlization,getJointParametersfromURDF
from scipy import signal
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from identification_numerics import differentiate_positions, scaled_least_squares

Order = [0,1,2,3,4,5,6]
import numpy as np

"""动力学参数辨识数值计算工具模块"""

# 执行加权最小二乘法进行参数估计，H: 回归矩阵，大小为 (m, n)  i: 电流数据，大小为 (m, 1)
def weighted_least_squares(H, i, max_iterations=100, tolerance=1e-6):
    return scaled_least_squares(H, i)[0]

# 通过Adaboost选择重要样本，A: 样本矩阵，M_fri: 摩擦矩阵，b: 样本对应的标签或目标值，n_samples: 选择的重要样本数量
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
    # 将重要样本转换回CasADi DM类型
    # A_important_cs = cs.DM(A_important)
    # b_important_cs = cs.DM(b_important)
    return A_important, M_fri_imp, b_important

# 动力学参数估计器
class Estimator():
    def __init__(self, node_name = "para_estimatior", dt_ = 5.0, N_ = 100, gravity_vec = [0.0, 0.0, -9.81]) -> None:
        self.dt_ = dt_
        self.model_ = "med7dock" #str(self.get_parameter("model").value)
        path = os.path.join(get_package_share_directory("med7_dock_description"), "urdf", f"{self.model_}.urdf.xacro",)
        self.N = N_
        # 获取机器人模型
        self.robot = optas.RobotModel(xacro_filename=path, time_derivs=[1],)
        # 获取机器人关节参数
        Nb, xyzs, rpys, axes = getJointParametersfromURDF(self.robot)
        # 计算机器人动力学模型
        self.dynamics_ = RNEA_function(Nb,1,rpys,xyzs,axes,gravity_para = cs.DM(gravity_vec))
        # 通过动态线性化方法获取动力学回归矩阵和参数向量
        self.Ymat, self.PIvector = DynamicLinearlization(self.dynamics_,Nb)

        urdf_string_ = xacro.process(path)
        robot = urdf.URDF.from_xml_string(urdf_string_)

        masses = [link.inertial.mass for link in robot.links if link.inertial is not None]#+[1.0]
        self.masses_np = np.array(masses[1:])
        # print("masses = {0}".format(self.masses_np))

        massesCenter = [link.inertial.origin.xyz for link in robot.links if link.inertial is not None]#+[[0.0,0.0,0.0]]
        self.massesCenter_np = np.array(massesCenter[1:]).T
        # Inertia = [np.mat(link.inertial.inertia.to_matrix()) for link in robot.links if link.inertial is not None]
        Inertia = [link.inertial.inertia.to_matrix() for link in robot.links if link.inertial is not None]
        self.Inertia_np = np.hstack(tuple(Inertia[1:]))
        
    @ staticmethod
    def readCsvToList(path):
        l = []
        with open(path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                joint_names = [x.strip() for x in list(row.keys())]
                l.append([float(x) for x in row.values()])
        return l    
    
    # 从csv
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
    
    def ExtractFromMeasurmentList(self, pos_list):
        dt = 0.01
        pos_l = []
        tau_ext_l = []
        # with open(path_pos) as csv_file:
        #     csv_reader = csv.DictReader(csv_file)
        for row in pos_list:
            # print("111 = {0}".format(row.values()))
            pl = row[0:7]
            tl = row[7:14]
            pos_l.append([float(x) for x in pl])
            tau_ext_l.append([float(x) for x in tl])
        vel_l, _ = differentiate_positions(pos_l, dt)
        return pos_l, vel_l.tolist(), tau_ext_l

    def ExtractFromMeasurmentListZeroVel(self,pos_list):
        dt = 0.01
        pos_l = []
        tau_ext_l = []
        # with open(path_pos) as csv_file:
        #     csv_reader = csv.DictReader(csv_file)
        for row in pos_list:
            # print("111 = {0}".format(row.values()))
            pl = row[0:7]
            tl = row[7:14]
            pos_l.append([float(x) for x in pl])
            tau_ext_l.append([float(x) for x in tl])
        vel_l =[]
        for id in range(len(pos_l)):
            vel_l.append([0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0])
        return pos_l,vel_l,tau_ext_l    
       
    # 保存
    def save_(self, csv_file, keys: List[str], values_list: List[List[float]]) -> None:
        csv_writer = csv.DictWriter(csv_file, fieldnames=keys)
        csv_writer.writeheader()
        for values in values_list:
            csv_writer.writerow({key: value for key, value in zip(keys,values)})
    
    def timer_cb_regressor_physical_con_impt_samp(self, positions, velocities, efforts):
        Pb, Pd, Kd =find_dyn_parm_deps(7,80,self.Ymat)
        K = Pb.T +Kd @Pd.T
        taus = []
        Y_ = []
        Y_fri = []
        # init_para = np.random.uniform(0.0, 0.1, size=50)
        # filter_list = [TD_2order(T=0.01) for i in range(7)]
        # filter_vector = TD_list_filter(T=0.01)
        if len(positions) < 2:
            raise ValueError("at least two samples are required")
        for k in range(1, len(positions)):
            # print("q_np = {0}".format(q_np))
            # q_np = np.random.uniform(-1.5, 1.5, size=7)
            q_np = [positions[k][i] for i in Order]
            # print("velocities[k] = {0}".format(velocities[k]))
            qd_np = [velocities[k][i] for i in Order]
            tau_ext = [efforts[k][i] for i in Order]
            qdlast_np = [velocities[k-1][i] for i in Order]
            qdd_np = (np.array(qd_np)-np.array(qdlast_np))/0.01
            qdd_np_list = qdd_np.tolist()
            Y_temp = self.Ymat(q_np, qd_np, qdd_np_list) @Pb 
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
        pa_size = Y_r.shape[1]
        taus1 = taus1.T
        # without friction
        # Y = Y_r #cs.DM(np.hstack((Y_r, Y_fri1)))
        # with friction
        Y = cs.DM(np.hstack((Y_r, Y_fri1)))
        # estimate_pam = np.linalg.inv(Y.T @ Y) @ Y.T @ taus1
        # print("self.masses_np",self.masses_np.shape)
        # print("self.masses_np",self.massesCenter_np.shape)
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
        # print("list_of_intertia_norminal = {0}".format(list_of_intertia_norminal[0]))
        ineq_constr += [I[0,0] <=I[1,1] +I[2,2] for I in list_of_intertia_norminal]
        ineq_constr += [I[1,1] <=I[0,0] +I[2,2] for I in list_of_intertia_norminal]
        ineq_constr += [I[2,2] <=I[1,1] +I[0,0] for I in list_of_intertia_norminal]
        ineq_constr += [100.0*cs.mmin(cs.vertcat(I[1,1], I[0,0], I[2,2]))  >=cs.mmax(cs.vertcat(I[1,1], I[0,0], I[2,2])) for I in list_of_intertia_norminal]
        # ineq_constr += [cs.trace(I)>0.0 for I in list_of_intertia_norminal]
        ineq_constr += [3.0 * list_of_intertia_norminal[j][2,2]<= cs.mmin(cs.vertcat(list_of_intertia_norminal[j][0,0], list_of_intertia_norminal[j][1,1])) for j in [0, 2, 4]]
        ineq_constr += [3.0 * list_of_intertia_norminal[k][1,1]<= cs.mmin(cs.vertcat(list_of_intertia_norminal[k][0,0], list_of_intertia_norminal[k][2,2])) for k in [1, 3]]
        ineq_constr += [1e-4<= I[0,0] for I in list_of_intertia_norminal]
        ineq_constr += [1e-4<= I[1,1] for I in list_of_intertia_norminal]
        ineq_constr += [1e-4<= I[2,2] for I in list_of_intertia_norminal]
        ineq_constr += [cs.mmax(cs.vertcat(cs.norm_2(I[1,0]), cs.norm_2(I[0,2]), cs.norm_2(I[1,2])))<= 0.1*cs.norm_2(cs.mmin(cs.vertcat(I[1,1], I[0,0], I[2,2]))) for I in list_of_intertia_norminal]
        # ineq_constr += [cs.norm_2(_estimate[_w0+i] - mass_center_norminal[i])> 0.1*cs.norm_2(mass_center_norminal[i]) for i in range(_w1*_h1)]
        # ineq_constr += [_estimate[i]> 0.0 for i in range(_w2*_h2)]
        problem = {'x': _estimate, 'f': obj, 'g': cs.vertcat(*ineq_constr)}
        # solver = cs.qpsol('solver', 'qpoases', problem)
        # solver = cs.nlpsol('S', 'ipopt', problem,{'ipopt':{'max_iter':3000000 }, 'verbose':True})
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
        # solver = cs.nlpsol('S', 'ipopt', problem,
        #               {'ipopt':{'max_iter':1000 }, 
        #                'verbose':False,
        #                "ipopt.hessian_approximation":"limited-memory"
        #                })
        print("solver = {0}".format(solver))
        # sol = S(x0 = init_x0,lbg = lbg, ubg = ubg)
        gt_x0 = mass_norminal.tolist()+mass_center_norminal.tolist()+intertia_norminal.tolist()+[0.1]*len(qd_np)+[0.5]*len(qd_np)
        import random
        init_x0 = (mass_norminal*np.random.uniform(1.5, 3.5, size=mass_norminal.shape)
            ).tolist()+(mass_center_norminal*np.random.uniform(0.0, 0.2, size=mass_center_norminal.shape)
                ).tolist()+(intertia_norminal*np.random.uniform(0.0, 0.1, size=intertia_norminal.shape)
                    ).tolist()+[random.random()*0.05 for _ in range(len(qd_np))]+[random.random()*0.2 for _ in range(len(qd_np))]
        # init_x0 = [random.randint(0, 100) for _ in range(len(gt_x0))]
        # sol = solver(x0 = [0.0]*len(init_x0))
        sol = solver(x0 = init_x0)
        # print("sol = {0}".format(sol['x']))
        # print("init_x0 = {0}".format(init_x0))
        # raise ValueError("run to here")
        preds = taus1 - Y_r @ e_cs_fun(sol['x']) -Y_fri1 @ sol['x'][-len(qd_np)*2:]
        Y_r1, Y_fri2,taus2 = select_important_samples(Y_r, Y_fri1,taus1, preds,140)
        print("Y_fri2 = ", Y_fri2.shape)
        obj2 = cs.sumsqr(taus2 - Y_r1 @ estimate_cs -Y_fri2 @ _estimate[-len(qd_np)*2:])+ 2.0 * cs.norm_2(_estimate[:_w0]) + 5.0 * cs.norm_2(_estimate[_w0:l1]) + 5.0 * cs.norm_2(_estimate[l1:l2])
        problem2 = {'x': _estimate, 'f': obj2, 'g': cs.vertcat(*ineq_constr)}
        solver2 = cs.nlpsol('S', 'ipopt', problem2, opts)
        sol2 = solver2(x0 = sol['x'])
        return sol2['x'],np.array(gt_x0)
    
    # 计算
    def get_Yb_matrix(self, positions, velocities, efforts,Pb):
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
        taus1 = np.hstack(taus).T
        Y_fri1 = np.vstack(Y_fri)
        return Y_r, taus1, Y_fri1
    
    # 构建不等式物理约束
    @staticmethod
    def build_ineq_physical_con(_estimate,
                                _w0, # max index of mass
                                _w1, # size1 of mass center 
                                _h1, # size2 of mass center
                                _w2, # size1 of inertia
                                _h2  # size2 of inertia
                                ):
        l1 = _w0 + _w1*_h1
        l2 = l1 + _w2 * _h2
        Inertia = _estimate[l1:l2].reshape((_w2,_h2))
        list_of_intertia_norminal = [Inertia[:, i:i+3] for i in range(0, Inertia.shape[1], 3)]
        constraints, lower, upper = [], [], []
        def bounded(expression, lb=-np.inf, ub=np.inf):
            constraints.append(expression)
            lower.append(lb)
            upper.append(ub)
        for i in range(_w0):
            bounded(_estimate[i], 1e-6, np.inf)
        for I in list_of_intertia_norminal:
            # The URDF inertia tensor is symmetric.  Explicit equality
            # constraints prevent the unused lower-triangular entries from
            # becoming arbitrary optimization variables.
            bounded(I[0, 1] - I[1, 0], 0.0, 0.0)
            bounded(I[0, 2] - I[2, 0], 0.0, 0.0)
            bounded(I[1, 2] - I[2, 1], 0.0, 0.0)
            # Sylvester criterion for a symmetric positive-definite inertia.
            bounded(I[0, 0], 1e-8, np.inf)
            bounded(I[0, 0] * I[1, 1] - I[0, 1] ** 2, 1e-12, np.inf)
            bounded(cs.det(I), 1e-15, np.inf)
            # Principal moments of a rigid body obey triangle inequalities.
            bounded(I[1, 1] + I[2, 2] - I[0, 0], 0.0, np.inf)
            bounded(I[0, 0] + I[2, 2] - I[1, 1], 0.0, np.inf)
            bounded(I[0, 0] + I[1, 1] - I[2, 2], 0.0, np.inf)
        # Coulomb and viscous friction magnitudes are non-negative in the
        # adopted sign(qd) + qd convention.
        for i in range(l2, _estimate.numel()):
            bounded(_estimate[i], 0.0, np.inf)
        return constraints, lower, upper
    
    # 
    @staticmethod
    def get_gt_params_sim(mass_norminal, mass_center_norminal, intertia_norminal, nj, fri_p1=0.1, fri_p2=0.5):
        # mass_norminal = self.masses_np
        # mass_center_norminal = self.massesCenter_np.reshape(-1,_w1*_h1).flatten()
        # intertia_norminal = self.Inertia_np.reshape(-1,_w2*_h2).flatten()
        gt_x0 = mass_norminal.tolist()+mass_center_norminal.tolist()+intertia_norminal.tolist()+[fri_p1]*nj+[fri_p2]*nj
        return gt_x0
    
    # 
    def get_gt_params_simO(self):
        nj = self.robot.ndof
        mass_norminal = self.masses_np
        _w1, _h1 =self.massesCenter_np.shape
        _w2, _h2 =self.Inertia_np.shape
        mass_center_norminal = self.massesCenter_np.reshape(-1,_w1*_h1).flatten()
        intertia_norminal = self.Inertia_np.reshape(-1,_w2*_h2).flatten()
        gt_x0 = Estimator.get_gt_params_sim(mass_norminal, mass_center_norminal, intertia_norminal, nj)
        return gt_x0

    # 通过优化求解器进行动力学参数辨识，考虑物理约束
    def timer_cb_regressor_physical_con(self, positions, velocities, efforts):
        nj = len(positions[0])
        # 获取动力学参数独立矩阵
        Pb, Pd, Kd =find_dyn_parm_deps(7,80,self.Ymat)
        K = Pb.T +Kd @Pd.T
        # 计算
        Y_r, taus1, Y_fri1 = self.get_Yb_matrix(positions, velocities, efforts, Pb)
        print("self.masses_np = ",self.masses_np)
        _w1, _h1 =self.massesCenter_np.shape
        _w2, _h2 =self.Inertia_np.shape
        _w0 = len(self.masses_np)
        l1 = _w0 + _w1*_h1
        l2 = l1 + _w2 * _h2
        # with friction
        l = l2+ nj*2
        _estimate = cs.SX.sym('para', l)
        estimate_cs = K @ self.PIvector(_estimate[0:_w0], _estimate[_w0:l1].reshape((_w1,_h1)), _estimate[l1:l2].reshape((_w2,_h2)))
        obj = cs.sumsqr(taus1 - Y_r @ estimate_cs -Y_fri1 @ _estimate[-nj*2:]) + 10.0 * cs.norm_2(_estimate[:_w0]) + 100.0 * cs.norm_2(_estimate[_w0:l1]) + 100.0 * cs.norm_2(_estimate[l1:l2])
        # Inertia = _estimate[l1:l2].reshape((_w2,_h2))
        # list_of_intertia_norminal = [Inertia[:, i:i+3] for i in range(0, Inertia.shape[1], 3)]
        ineq_constr, constraint_lb, constraint_ub = Estimator.build_ineq_physical_con(_estimate, _w0, _w1, _h1, _w2, _h2)
        problem = {'x': _estimate, 'f': obj, 'g': cs.vertcat(*ineq_constr)}
        # solver = cs.qpsol('solver', 'qpoases', problem)
        # solver = cs.nlpsol('S', 'ipopt', problem,{'ipopt':{'max_iter':3000000 }, 'verbose':True})
        opts = {
            'ipopt': {
                'max_iter': 5000,  # 提高最大迭代次数
                'tol': 1e-10,  # 更严格的容忍度
                'constr_viol_tol': 1e-9,  # 约束违反容差
                'compl_inf_tol': 1e-9,  # 互补性条件容差
                'acceptable_tol': 1e-8,  # 更严格的可接受容忍度
                'acceptable_iter': 20,  # 提高可接受的最大迭代次数
                'linear_solver': 'mumps',  # 或 'ma57', 'mumps'，选择最适合的求解器
                'mu_strategy': 'adaptive',  # 自适应 mu 策略
                'dual_inf_tol': 1e-10,  # 更严格的对偶可行性容忍度
                'compl_inf_tol': 1e-10,  # 更严格的互补性容忍度
                'bound_relax_factor': 0,  # 防止约束松弛
                'hessian_approximation': 'exact',  # 使用精确的 Hessian，不使用近似
            },
            'verbose': False,  # 如果需要调试信息，可以设置为 True
        }
        # 创建求解器
        solver = cs.nlpsol('S', 'ipopt', problem, opts)
        # solver = cs.nlpsol('S', 'ipopt', problem,
        #               {'ipopt':{'max_iter':1000 }, 
        #                'verbose':False,
        #                "ipopt.hessian_approximation":"limited-memory"
        #                })
        print("solver = {0}".format(solver))
        mass_norminal = self.masses_np
        mass_center_norminal = self.massesCenter_np.reshape(-1,_w1*_h1).flatten()
        intertia_norminal = self.Inertia_np.reshape(-1,_w2*_h2).flatten()
        
        gt_x0 = Estimator.get_gt_params_sim(mass_norminal, mass_center_norminal, intertia_norminal, nj)
        # gt_x0 = mass_norminal.tolist()+mass_center_norminal.tolist()+intertia_norminal.tolist()+[0.1]*nj+[0.5]*nj
        # init_x0 = [random.randint(0, 10) for _ in range(len(gt_x0))]
        import random
        init_x0 = (mass_norminal*np.random.uniform(0.0, 2.0, size=mass_norminal.shape)).tolist()+(mass_center_norminal*np.random.uniform(0.0, 2.0, size=mass_center_norminal.shape)
                ).tolist()+(intertia_norminal*np.random.uniform(0.0, 2.0, size=intertia_norminal.shape)).tolist()+[random.random()*1.0 for _ in range(nj)]+[random.random()*1.0 for _ in range(nj)]
        # sol = solver(x0 = [0.0]*len(init_x0))
        sol = solver(x0=init_x0, lbg=constraint_lb, ubg=constraint_ub)
        stats = solver.stats()
        if not stats.get('success', False):
            raise RuntimeError("physical parameter optimization failed: " + str(stats.get('return_status', 'unknown status')))
        return sol['x'],np.array(gt_x0)
    
    # 使用最小二乘法估计动力学参数
    def timer_cb_regressor(self, positions, velocities, efforts):
        # 获取动力学参数独立性矩阵
        Pb, Pd, Kd =find_dyn_parm_deps(7,80,self.Ymat)
        K = Pb.T +Kd @Pd.T
        # q_nps = []
        # qd_nps = []
        # qdd_nps = []
        taus = []
        Y_ = []
        Y_fri = []
        # init_para = np.random.uniform(0.0, 0.1, size=50)
        # filter_list = [TD_2order(T=0.01) for i in range(7)]
        # filter_vector = TD_list_filter(T=0.01)
        for k in range(1, len(positions)):
            # print("q_np = {0}".format(q_np))
            # q_np = np.random.uniform(-1.5, 1.5, size=7)
            q_np = [positions[k][i] for i in Order]
            # print("velocities[k] = {0}".format(velocities[k]))
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
        # solver = cs.qpsol('solver', 'qpoases', problem)
        solver = cs.nlpsol('S', 'ipopt', problem,{'ipopt':{'max_iter':3000000 }, 'verbose':True})
        print("solver = {0}".format(solver))
        sol = solver()
        print("sol = {0}".format(sol['x']))
        return sol['x'],estimate_pam
    
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
        # _estimate = cs.SX.sym('para', l)
        estimate_cs = K @ self.PIvector(para[0:_w0], para[_w0:l1].reshape((_w1,_h1)), para[l1:l].reshape((_w2,_h2)))
        estimate_gt = K @ self.PIvector(para_gt[0:_w0], para_gt[_w0:l1].reshape((_w1,_h1)), para_gt[l1:l].reshape((_w2,_h2)))
        for k in range(1,len(positions),1):
            q_np = [positions[k][i] for i in Order]
            qd_np = [velocities[k][i] for i in Order]
            # tau_ext = [efforts[k][i] for i in Order]
            qdlast_np = [velocities[k-1][i] for i in Order]
            qdd_np = (np.array(qd_np)-np.array(qdlast_np))/0.01#(velocities[k][0]-velocities[k-1][0])
            # qdd_np = [f(qd_np[id])[1] for id,f in enumerate(filter_list)]
            pa_size = Pb.shape[1]
            tau_est_model = (self.Ymat(q_np,qd_np,qdd_np) @Pb@  estimate_cs + np.diag(np.sign(qd_np)) @ para[-2*len(qd_np):-len(qd_np)] + np.diag(qd_np) @ para[-len(qd_np):])
            tau_ext = (self.Ymat(q_np,qd_np,qdd_np) @Pb@  estimate_gt + np.diag(np.sign(qd_np)) @ para_gt[-2*len(qd_np):-len(qd_np)] + np.diag(qd_np) @ para_gt[-len(qd_np):])
            # tau_est_model = (self.Ymat(q_np,qd_np,qdd_np) @Pb@  estimate_cs )
            e= tau_est_model - tau_ext 
            print("sim_tau = {0}".format(tau_ext))
            print("tau_est_model = {0}".format(tau_est_model))
            print("sim_tau 2 = {0}".format(self.dynamics_(q_np,qd_np, qdd_np, self.masses_np, para_gt[_w0:l1].reshape((_w1,_h1)), para_gt[l1:l].reshape((_w2,_h2)))))
            # print("tau_error = {0}".format(e))
            # print("q_np = {0}".format(q_np))
            tau_ests.append(tau_est_model.toarray().flatten().tolist())
            es.append(e.toarray().flatten().tolist())
            tau_exts.append(tau_ext.toarray().flatten().tolist())
        return tau_ests, tau_exts
    
    # 使用估计的参数计算的关节力矩与实际测量的关节力矩之间的误差
    def testWithEstimatedParaCon(self, positions, velocities, efforts, para)->None:
        # 获取动力学参数独立性矩阵
        Pb, Pd, Kd =find_dyn_parm_deps(7,80,self.Ymat)
        K = Pb.T +Kd @Pd.T
        tau_ests = []
        es = []
        # 使用二阶低通滤波器对速度进行滤波
        filter_list = [TD_2order(T=0.01) for i in range(7)]
        _w1, _h1 =self.massesCenter_np.shape
        _w2, _h2 =self.Inertia_np.shape
        _w0 = len(self.masses_np)
        l = _w0 + _h1*_w1 + _w2 * _h2
        l1 = _w0 + _w1*_h1
        estimate_cs = K @ self.PIvector(para[0:_w0], para[_w0:l1].reshape((_w1,_h1)), para[l1:l].reshape((_w2,_h2)))
        for k in range(1,len(positions),1):
            q_np = [positions[k][i] for i in Order]
            qd_np = [velocities[k][i] for i in Order]
            tau_ext = [efforts[k][i] for i in Order]
            qdlast_np = [velocities[k-1][i] for i in Order]
            qdd_np = (np.array(qd_np)-np.array(qdlast_np))/0.01#(velocities[k][0]-velocities[k-1][0])
            # qdd_np = [f(qd_np[id])[1] for id,f in enumerate(filter_list)]
            pa_size = Pb.shape[1]
            tau_est_model = (self.Ymat(q_np,qd_np,qdd_np) @Pb@  estimate_cs + np.diag(np.sign(qd_np)) @ para[-2*len(qd_np):-len(qd_np)] + np.diag(qd_np) @ para[-len(qd_np):])
            # tau_est_model = (self.Ymat(q_np,qd_np,qdd_np) @Pb@  estimate_cs )
            e= tau_est_model - tau_ext 
            print("sim_tau = {0}".format(tau_ext))
            print("tau_est_model = {0}".format(tau_est_model))
            # print("tau_error = {0}".format(e))
            print("q_np = {0}".format(q_np))
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
            # q_np = positions[k][4,1,2,3,5,6,7]
            # qd_np = velocities[k][4,1,2,3,5,6,7]
            # tau_ext = efforts[k][4,1,2,3,5,6,7]
            # qdd_np = (np.array(velocities[k][4,1,2,3,5,6,7])-np.array(velocities[k-1][4,1,2,3,5,6,7]))/(velocities[k][0]-velocities[k-1][0])
            # qdd_np = qdd_np.tolist()
            q_np = [positions[k][i] for i in Order]
            qd_np = [velocities[k][i] for i in Order]
            tau_ext = [efforts[k][i] for i in Order]
            qdlast_np = [velocities[k-1][i] for i in Order]
            qdd_np = (np.array(qd_np) - np.array(qdlast_np))/0.01   #(velocities[k][0]-velocities[k-1][0])
            # qdd_np = qdd_np.tolist()
            # qdd_np = (np.array(qd_np)-np.array(qdlast_np))/0.01
            # qdd_np = qdd_np.tolist()
            qdd_np = [f(qd_np[id])[1] for id, f in enumerate(filter_list)]
            # tau_ext = self.robot.rnea(q_np,qd_np,qdd_np)
            # e=self.Ymat(q_np,qd_np,qdd_np)@Pb @ (solution[f"{self.pam_name}/y"] -  K @real_pam)
            # print("error = {0}".format(e))
            # e=self.Ymat(q_np,qd_np,qdd_np)@Pb @  para - tau_ext 
            pa_size = Pb.shape[1]
            tau_est_model = (self.Ymat(q_np,qd_np,qdd_np) @Pb@  para[:pa_size] + np.diag(np.sign(qd_np)) @ para[pa_size:pa_size+7] + np.diag(qd_np) @ para[pa_size+7:])
            # without friction
            # tau_est_model = (self.Ymat(q_np,qd_np,qdd_np) @Pb@  para[:pa_size] )
            e= tau_est_model - tau_ext 
            print("error1 = {0}".format(e))
            print("tau_ext = {0}".format(tau_ext))
            tau_ests.append(tau_est_model.toarray().flatten().tolist())
            es.append(e.toarray().flatten().tolist())
        return tau_ests, es

    # 将估计的参数保存到CSV文件中
    def saveEstimatedPara(self, parac)->None:
        path1 = os.path.join(get_package_share_directory("gravity_compensation"), "test", "DynamicParameters.csv",)
        para = parac.toarray().flatten()
        keys = ["para_{0}".format(idx) for idx in range(len(para))]
        with open(path1,"w") as csv_file:
            self.save_(csv_file,keys, [para])
            
# 使用butterworth滤波器对轨迹数据进行低通滤波
def traj_filter(states):
    cols = []
    l=len(states[0])
    fs = 100
    cutoff_freq = 2  # 截止频率为10 Hz
    b, a = signal.butter(4, cutoff_freq / (fs / 2), 'low')
    filtered_signal = []
    states_filtered = []
    for i in range(l):
        cols.append([float(state[i]) for state in states])
        filtered_signal.append( signal.filtfilt(b, a, cols[i]))
    for j in range(len(filtered_signal[0])):
        states_filtered.append([filtered_signal[i][j] for i in range(l)])
    return states_filtered

# 比较轨迹，绘制估计的外部力和实际的外部力
def compare_traj(states1, states2):
    col1s , col2s = [], []
    l=len(states1[0])
    fig, axs = plt.subplots(7, 1, figsize=(8,10))
    for i in range(l):
        print("states = {0}".format(states2[i]))
        col1s.append([float(state[i]) for state in states1])
        col2s.append([float(state[i]) for state in states2])
        axs[i].plot(col1s[i])
        axs[i].plot(col2s[i])
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def main(args=None):
    rclpy.init(args=args)
    # 获取机器人参数估计器
    paraEstimator = Estimator()
    # 获取数据
    path_pos = os.path.join(get_package_share_directory("gravity_compensation"), "test", "robot_data copy 2.csv", )
    # 从CSV文件中提取位置、速度和努力数据
    positions, velocities, efforts = paraEstimator.ExtractFromMeasurmentCsv(path_pos)
    velocities=traj_filter(velocities)
    efforts_f=traj_filter(efforts)
    # 进行参数估计
    estimate_pam, ref_pam = paraEstimator.timer_cb_regressor_physical_con(positions, velocities, efforts_f)
    print("estimate_pam = {0}".format(estimate_pam))
    # 进行测试，使用估计的参数进行控制
    tau_exts, es =paraEstimator.testWithEstimatedParaCon(positions, velocities, efforts_f,estimate_pam)
    # 保存估计的参数到CSV文件
    paraEstimator.saveEstimatedPara(estimate_pam)
    # 比较轨迹，绘制估计的外部力和实际的外部力
    compare_traj(tau_exts, efforts_f)

    path_pos_2 = os.path.join(get_package_share_directory("gravity_compensation"), "test", "measurements_0dgr.csv", )
    # 从CSV文件中提取位置、速度和努力数据
    positions_, velocities_, efforts_ = paraEstimator.ExtractFromMeasurmentCsv(path_pos_2)
    velocities_=traj_filter(velocities_)
    efforts_f_=traj_filter(efforts_)
    # 进行测试，使用估计的参数进行控制
    tau_exts_, es =paraEstimator.testWithEstimatedParaCon(positions_, velocities_, efforts_f_,estimate_pam)
    # 比较轨迹，绘制估计的外部力和实际的外部力
    compare_traj(tau_exts_, efforts_f_)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
