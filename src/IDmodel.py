import csv
import numpy as np
import optas
from optas.spatialmath import *
from optas.spatialmath import rpy2r, angvec2r, skew
import casadi as cs

from ament_index_python import get_package_share_directory
import os
import math
import xacro
import urdf_parser_py.urdf as urdf
from identification_numerics import base_parameter_transform

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

# 根据位置误差x1-u 和当前速度x2，计算下一步应该施加的“加速度”，使x1快速且平稳地跟踪u。r: 速度/加速度强度 h: 滤波步长
def fhan(x1, x2, u, r, h):
    d = r * h
    d0 = d * h
    # 计算预测位置与目标位置误差
    y = x1 - u + h * x2
    # 根据最速控制律计算加速度
    a0 = math.sqrt(d * d + 8 * r * abs(y))
    if abs(y) <= d0:
        a = x2 + y/h
    else:
        a = x2 + 0.5 * (a0 - d) * sign(y)
    if abs(a)<=d:
        return -r * a/d
    else:
        return -r * sign(a)

# 离散二阶跟踪微分器。目标u变化时，产生一个平滑跟踪信号x1，同时由x2给出该信号的近似导数
class TD_2order:
    def __init__(self, T=0.01, r=10.0, h=0.1):
        self.x1 = None
        self.x2 = None
        self.T = T
        self.r = r
        self.h = h
    def __call__(self, u):
        if self.x1 is None or self.x2 is None:
            self.x1 = 0
            self.x2 = 0
        x1k = self.x1
        x2k = self.x2
        self.x1 = x1k + self.T* x2k
        self.x2 = x2k + self.T* fhan(x1k, x2k, u, self.r, self.h)
        return self.x1, self.x2
    
# 同时对一组输入分别运行二阶跟踪微分器
class TD_list_filter:
    def __init__(self, T=0.01, r=10.0, h=0.1, len = 7) -> None:
        self.x1_list = None
        self.x2_list = None
        self.T = T
        self.r = r
        self.h = h
        self.len = len
    def __call__(self, us):
        if self.x1_list is None or self.x2_list is None:
            self.x1_list = [0.0] * self.len
            self.x2_list = [0.0] * self.len
        x1k = np.array(self.x1_list)
        x2k = np.array(self.x2_list)
        self.x1_list = x1k + self.T *x2k
        f = np.array([fhan(x1, x2, u, self.r, self.h) for (x1, x2, u) in zip(x1k, x2k, us)])
        self.x2_list = x2k + self.T *f
        return self.x1_list, self.x2_list

# 从csv文件读取参数
def ExtractFromParamsCsv(path):
        params = []
        with open(path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                params = [float(x) for x in list(row.values())]
        return params

# 递归牛顿欧拉算法。Nb: 关节数  Nk: 额外末端固定结构数量(基本为1) 
# rpys[i]: 连杆i的坐标系相对连杆i-1的坐标系的变换，连杆0是第一个连杆不是root，大小为NfX3，(0、...、Nf-1)，包括末端刚体相对最后连杆的旋转矩阵
# xyzs[i]: 连杆i的原点相对连杆i-1的原点的平移，连杆0是第一个连杆不是root，大小为NfX3，(0、...、Nf-1)，包括末端刚体相对最后连杆的坐标系平移
# axes[i]: 连杆i的旋转轴，连杆0是第一个连杆不是root，大小为NfX3，(0、...、Nf-1)，包括末端刚体相对最后连杆的连接轴
# gravity_para: 基坐标系下的重力加速度
def RNEA_function(Nb, Nk, rpys, xyzs, axes, gravity_para = cs.DM([0, 0, -9.81])):
    Nf = Nb+Nk
    om0 = cs.DM([0.0,0.0,0.0])
    om0D = cs.DM([0.0,0.0,0.0])
    # 关节位置、速度、加速度符号变量
    q = cs.SX.sym('q', Nb, 1)
    qd = cs.SX.sym('qd', Nb, 1)
    qdd = cs.SX.sym('qdd', Nb, 1)
    # 连杆1到末端执行器的质量、质心、惯性矩阵
    m = cs.SX.sym('m', 1, Nb+1)
    cm = cs.SX.sym('cm', 3, Nb+1)
    Icm = cs.SX.sym('Icm', 3, 3*Nb+3)
    # 合力和合力矩列表
    fs = [cs.DM([0.0,0.0,0.0])]
    ns = [cs.DM([0.0,0.0,0.0])]
    # 各关节坐标系的角速度、角加速度、线加速度
    oms = [om0]
    omDs = [om0D]
    vDs = [-gravity_para]
    # 正向推导各关节的角速度、角加速度、线加速度
    for i in range(Nf): # i从0开始循环到Nf-1
        if(i != Nf-1):
            # 计算i-1相对i的旋转矩阵
            iRp = (rpy2r(rpys[i]) @ angvec2r(q[i], axes[i])).T
            iaxisi = iRp @ axes[i]
            # 计算连杆i的角速度: 上个连杆的角速度 + 当前连杆绕关节的旋转角速度
            omi = iRp @ oms[i] + iaxisi * qd[i]
            # 计算连杆i的角加速度: 上个连杆的角加速度 + 上个连杆角速度 X 当前连杆角速度 + 当前关节的角加速度
            omDi = iRp @ omDs[i] +  skew(iRp @ oms[i]) @ (iaxisi*qd[i]) + iaxisi*qdd[i]
        else:
            # 末端执行器没有可动关节，所以只有rpy
            iRp = rpy2r(rpys[i]).T
            # 末端执行器的加速度和角加速度只需要转换一下上个关节的坐标系
            omi = iRp @ oms[i]
            omDi = iRp @ omDs[i]
        # 计算连杆i的线加速度: 变换矩阵 X (上个连杆的线加速度 + 上个连杆的角加速度 X 半径 + 上个连杆的角速度计算的向心加速度)
        vDi = iRp @ (vDs[i] + skew(omDs[i]) @ xyzs[i] + skew(oms[i]) @ (skew(oms[i]) @ xyzs[i]))
        # 计算连杆i的合力: 质量 * (当前连杆的线加速度 + 当前连杆的角加速度 X 质心向量 + 当前连杆的角速度 X (当前连杆的角速度 X 质心向量)) 
        fi = m[i] * (vDi + skew(omDi) @ cm[:,i] + skew(omi) @ (skew(omi) @ cm[:,i]))
        # 计算连杆i的合力矩: 当前连杆的惯性矩阵 X 当前连杆的角加速度 + 陀螺力矩
        ni = Icm[:, i*3: i*3+3] @ omDi + skew(omi) @ Icm[: ,i*3: i*3+3] @ omi 
        # 保存到列表，矩阵大小为3X(Nf+1)，(基座、连杆1...连杆Nb、末端刚体)，(0、...、Nf)
        oms.append(omi)
        omDs.append(omDi)
        vDs.append(vDi)
        fs.append(fi)
        ns.append(ni)
    # 反向推导各个关节的力矩
    ifi = fs[-1]
    # 把末端刚体关于质心的惯性力矩转换为关于坐标系原点的惯性力矩: 关于质心的惯性力矩 + 质心向量 X 惯性合力
    ini = ns[-1] + skew(cm[:,-1]) @ fs[-1]
    taus = []
    for i in range(Nf-1, 0, -1): # Nf-1到1，遍历活动关节求力矩，不包括末端刚体与最后一个关节的连接
        if(i < Nf-1):
            # 计算i相对i-1的旋转矩阵
            pRi = rpy2r(rpys[i]) @ angvec2r(q[i], axes[i])
        elif(i == Nf-1):
            # 末端没有旋转轴
            pRi = rpy2r(rpys[i])
        else:
            pRi = rpy2r(rpys[i])
        # 计算关节i-1施加给关节i的力矩: 惯性合力矩 + 关节i施加给关节i+1的力矩 + 惯性合力对坐标系原点产生的力矩 + 上个关节的力产生的力矩
        ini = ns[i] + pRi @ ini + skew(cm[:, i-1]) @ fs[i] + skew(xyzs[i]) @ pRi @ ifi
        # 计算关节i-1施加给关节i的力: 惯性合力 + 当前关节施加给上个关节的力
        ifi = fs[i] + pRi @ ifi
        # 计算i-1相对i-2的旋转矩阵
        pRi = rpy2r(rpys[i-1]) @ angvec2r(q[i-1], axes[i-1])
        # 计算作用在旋转轴的力矩: 
        _tau = ini.T @ pRi.T @ axes[i-1]
        taus.append(_tau)
    tau_ = cs.vertcat(*[taus[k] for k in range(len(taus)-1, -1, -1)])
    dynamics_ = optas.Function('dynamics', [q, qd, qdd, m, cm, Icm], [tau_])
    return dynamics_

# 把动力学方程写成对惯性参数线性的形式。 Nb: 主动关节数量
def DynamicLinearlization(dynamics_, Nb):
    # 关节位置、速度、加速度符号变量
    q = cs.SX.sym('q', Nb, 1)
    qd = cs.SX.sym('qd', Nb, 1)
    qdd = cs.SX.sym('qdd', Nb, 1)
    # 连杆1到末端执行器的质量、质心、惯性矩阵符号变量
    m = cs.SX.sym('m', 1, Nb+1)
    cm = cs.SX.sym('cm', 3, Nb+1)
    Icm = cs.SX.sym('Icm', 3, 3*Nb+3)
    # 计算回归矩阵
    Y = []
    for i in range(Nb): # 0到Nb-1，遍历每个关节的力矩表达式，dynamics_函数求出的
        Y_line = []
        for j in range(m.shape[1]): # 0到Nb，遍历每个连杆包含末端刚体
            # 提取质量参数系数
            m_indu = np.zeros([m.shape[1], m.shape[0]])                     # Nb+1行，1列
            cm_indu = np.zeros([3, Nb+1])                                   # 3行，Nb+1列    
            Icm_indu = np.zeros([3, 3*Nb+3])                                # 3行，3(Nb+1)列      
            m_indu[j] = 1.0                                                 # 只有第j行质量不为0，只保留第j行质量系数，表示第j个连杆的质量系数
            output = dynamics_(q, qd, qdd, m_indu, cm_indu, Icm_indu)[i]    # 惯性系数只有第j个连杆的质量不为0且为1，所以output是第j个连杆的质量系数
            Y_line.append(output)
            # 提取质心系数
            output1 = dynamics_(q, qd, qdd, m_indu, cm, Icm_indu)[i] - output # 惯性系数只有第j个连杆的质量和所有连杆质心向量不为0，质心向量都和质量都是成对存在，所以output1是第j个连杆的质心项
            for k in range(3):
                # 对第j个刚体的3个质心变量cm[k,j]求导，只保留质心系数
                output_cm = optas.jacobian(output1, cm[k,j])
                # 在符号表达式中进行变量替换，把cm替换成cm_indu，清理掉质心二次项求导后剩余的符号变量，只保留一次项系数，二次项系数在下面的惯性参数pi_temp里表示
                output_cm1 = optas.substitute(output_cm, cm, cm_indu) 
                Y_line.append(output_cm1)
            # 提取惯性参数系数
            output2 = dynamics_(q, qd, qdd, m_indu, cm_indu, Icm)[i] - output # 只保留惯性项
            o = 3 * j
            # 对第j个刚体的6个惯性参数求导，这里不包括二次项所以不用替换
            Y_line.extend([optas.jacobian(output2, Icm[0, o]),                                           # Ixx
                           optas.jacobian(output2, Icm[0, o+1]) + optas.jacobian(output2, Icm[1, o]),    # Ixy
                           optas.jacobian(output2, Icm[0, o+2]) + optas.jacobian(output2, Icm[2, o]),    # Ixz 
                           optas.jacobian(output2, Icm[1, o+1]),                                         # Iyy
                           optas.jacobian(output2, Icm[1, o+2]) + optas.jacobian(output2, Icm[2, o+1]),  # Iyz 
                           optas.jacobian(output2, Icm[2, o+2]),                                         # Izz
            ])
        # 水平拼接为一行
        sx_lst = optas.horzcat(*Y_line)
        Y.append(sx_lst)
    # 将各行竖直拼接为矩阵
    Y_mat = optas.vertcat(*Y)
    Ymat = optas.Function('Dynamic_Ymat', [q, qd, qdd], [Y_mat])
    # 计算PI向量
    PI_a = []
    for j in range(m.shape[1]):
        # 每个连杆10维的参数向量
        pi_temp = [m[j],                                                         # 质量
                   m[j] * cm[0,j],                                               # 质心向量x
                   m[j] * cm[1,j],                                               # 质心向量y
                   m[j] * cm[2,j],                                               # 质心向量z
                   Icm[0, 0+3*j] + m[j] * (cm[1,j]*cm[1,j] + cm[2,j]*cm[2,j]),   # XXi
                   Icm[0, 1+3*j] - m[j] * (cm[0,j]*cm[1,j]),                     # XYi
                   Icm[0, 2+3*j] - m[j] * (cm[0,j]*cm[2,j]),                     # XZi
                   Icm[1, 1+3*j] + m[j] * (cm[0,j]*cm[0,j] + cm[2,j]*cm[2,j]),   # YYi
                   Icm[1, 2+3*j] - m[j] * (cm[1,j]*cm[2,j]),                     # YZi
                   Icm[2, 2+3*j] + m[j] * (cm[0,j]*cm[0,j] + cm[1,j]*cm[1,j])]   # ZZi
        # 把pi_temp竖直拼接为一列
        PI_a.append(optas.vertcat(*pi_temp))
    # 将各列竖直拼接为一长列
    PI_vecter = optas.vertcat(*PI_a)
    # 把参数向量符号表达式封装成函数
    PIvector = optas.Function('Dynamic_PIvector', [m, cm, Icm], [PI_vecter])
    return Ymat, PIvector

# 通过随机采样回归矩阵并做奇异值分解，判断动力学参数之间是否存在线性依赖
def find_eigen_value(dof, parm_num, regressor_func, shape):
    samples = 100
    A_mat = np.zeros((shape, shape))
    for i in range(samples):
        a = np.random.random([parm_num, dof])*2.0 - 1.0
        b = np.random.random([parm_num, dof])*2.0 - 1.0
        A_mat = A_mat + regressor_func(a, b)
    U, s, V = np.linalg.svd(A_mat)
    return U, V

# 从urdf文件获取关节参数
def getJointParametersfromURDF(robot, ee_link="link_ee"):
    robot_urdf = robot.urdf
    root = robot_urdf.get_root()
    xyzs, rpys, axes = [], [], []
    print("link_names = ",robot.link_names)
    # 从root连杆到ee连杆
    joints_list = robot_urdf.get_chain(root, ee_link, links=False)
    # 提取xyzs、rpys、axes
    joints_list_r = joints_list[1:]
    for joint_name in joints_list_r:
        joint = robot_urdf.joint_map[joint_name]
        xyz, rpy = robot.get_joint_origin(joint)
        axis = robot.get_joint_axis(joint)
        # 保存运动学参数
        xyzs.append(xyz)
        rpys.append(rpy)
        axes.append(axis) # 相对于所在关节的坐标系
    Nb = len(joints_list_r)-1
    return Nb, xyzs, rpys, axes

# 通过随机采样，寻找动力学参数之间的线性依赖关系
def find_dyn_parm_deps(dof, parm_num, regressor_func, samples=10000, seed=0, rtol=None):
    rng = np.random.default_rng(seed)
    Z = np.zeros((dof * samples, parm_num))
    for i in range(samples):
        q = rng.uniform(-np.pi, np.pi, dof)
        dq = rng.uniform(-np.pi, np.pi, dof)
        ddq = rng.uniform(-np.pi, np.pi, dof)
        Z[i * dof : (i + 1) * dof, :] = np.asarray(regressor_func(q, dq, ddq)).reshape(dof, parm_num)
    # 得到基础参数Pb(独立可辨识参数)、从属参数Pd(可由基础参数线性表示)、从属参数到基础参数的转换关系Kd
    Pb, Pd, Kd, _ = base_parameter_transform(Z, rtol=rtol)
    return np.asmatrix(Pb), np.asmatrix(Pd), np.asmatrix(Kd)

# 从URDF/Xacro机器人模型中读取各连杆的质量、质心、惯性参数
def InertialParaFromURDF(path):
    urdf_string_ = xacro.process(path)
    robot = urdf.URDF.from_xml_string(urdf_string_)
    masses = [link.inertial.mass for link in robot.links if link.inertial is not None]#+[1.0]
    masses_np = np.array(masses[1:])
    massesCenter = [link.inertial.origin.xyz for link in robot.links if link.inertial is not None]#+[[0.0,0.0,0.0]]
    massesCenter_np = np.array(massesCenter[1:]).T
    Inertia = [link.inertial.inertia.to_matrix() for link in robot.links if link.inertial is not None]
    Inertia_np = np.hstack(tuple(Inertia[1:]))
    return masses_np, massesCenter_np, Inertia_np

def main():
    path = os.path.join(get_package_share_directory("med7_dock_description"), "urdf", "med7dock.urdf.xacro",)
    masses_np, massesCenter_np, Inertia_np = InertialParaFromURDF(path)
    robot = optas.RobotModel(xacro_filename=path, time_derivs=[1],)
    Nb, xyzs, rpys, axes = getJointParametersfromURDF(robot)
    dynamics_ = RNEA_function(Nb, 1, rpys, xyzs, axes)
    Ymat, PIvector = DynamicLinearlization(dynamics_, Nb)
    Pb, Pd, Kd = find_dyn_parm_deps(7, 80, Ymat)
    K = Pb.T + Kd @ Pd.T
    pa_size = Pb.shape[1]
    # 生成轨迹
    q = np.array([1.0]*7)
    qd = np.array([0.0]*7)
    qdd = np.array([0.0]*7)
    filter = TD_list_filter(T = 0.01)
    # 加载动力学参数
    path_pos = os.path.join(get_package_share_directory("gravity_compensation"), "test", "DynamicParameters.csv", )
    params = ExtractFromParamsCsv(path_pos)
    # 从urdf得到的惯性参数
    real_pam = PIvector(masses_np, massesCenter_np, Inertia_np)
    # 估计力矩
    tau_est = (Ymat(q.tolist(), filter(qd.tolist())[0], filter(qd.tolist())[1]) @ Pb  @  params[:pa_size] + np.diag(np.sign(qd)) @ params[pa_size:pa_size+7] + np.diag(qdd) @ params[pa_size+7:])
    print(" The estimated torque  tau_est = {0}".format(tau_est))

if __name__ == "__main__":
    main()


