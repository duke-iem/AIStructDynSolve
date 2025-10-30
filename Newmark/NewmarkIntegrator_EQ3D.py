"""
AIStructDynSolve
A framework focused on solving structural dynamics problems using artificial intelligence (AI) methods
solve the following ODE of MDOF
M*U_dotdot+C*U_dot+K*U=Pt
initial condition: U(t=0)=InitialU, and U_dot(t=0)=InitialU_dot
Author: 杜轲 duke@iem.ac.cn
Date: 2025/03/18
"""

import numpy as np
import pandas as pd

class NewmarkIntegrator_EQ3D:
    def __new__(cls, M, C, K, eigenvectors, time, EQX, EQY, EQZ, InitialU=None, InitialU_dot=None, integration_type='linear', Deltatime =0.01):
        """
        初始化并直接计算结果。

        参数:
        - M (array): 质量矩阵。
        - C (array): 阻尼矩阵。
        - K (array): 刚度矩阵。
        - eigenvectors (array): 特征向量矩阵。
        - time (array): 时间向量。
        - EQ (array): 地面加速度。
        - InitialU (array, optional): 初始位移。如果未提供，默认值为 0。
        - InitialU_dot (array, optional): 初始速度。如果未提供，默认值为 0。
        - integration_type (str): 积分类型 ('linear' 或 'average')。
        """
        instance = super(NewmarkIntegrator_EQ3D, cls).__new__(cls)

        # 初始化参数
        if isinstance(M, np.ndarray):
            instance.M = M
        else:
            instance.M = M.numpy()

        if isinstance(C, np.ndarray):
            instance.C = C
        else:
            instance.C = C.numpy()

        if isinstance(K, np.ndarray):
            instance.K = K
        else:
            instance.K = K.numpy()

        instance.eigenvectors = eigenvectors
        if isinstance(time, pd.Series):
            time_end = time.iloc[-1]
        else:
            time_end = time[-1]
        instance.time = np.arange(0, time_end+Deltatime, Deltatime)
        instance.EQX = np.interp(instance.time, time, EQX)
        instance.EQY = np.interp(instance.time, time, EQY)
        instance.EQZ = np.interp(instance.time, time, EQZ)
        instance.integration_type = integration_type

        # 如果没有提供 InitialU 和 InitialU_dot，初始化为 0.0
        instance.InitialU = np.zeros(instance.M.shape[0]) if InitialU is None else InitialU
        instance.InitialU_dot = np.zeros(instance.M.shape[0]) if InitialU_dot is None else InitialU_dot

        # 根据积分类型选择参数
        if instance.integration_type.lower() == 'linear':
            instance.gam, instance.bta = 0.5, 1 / 6
        elif instance.integration_type.lower() == 'average':
            instance.gam, instance.bta = 0.5, 1 / 4
        else:
            raise ValueError("未知的积分类型。使用 'linear' 或 'average'。")

        # 计算模态矩阵
        instance.M_modal = instance.eigenvectors.T @ instance.M @ instance.eigenvectors
        instance.C_modal = instance.eigenvectors.T @ instance.C @ instance.eigenvectors
        instance.K_modal = instance.eigenvectors.T @ instance.K @ instance.eigenvectors
        instance.n, instance.N = instance.M.shape[0], len(instance.time)

        # 初始计算
        # 计算EQ矩阵
        EQ_vec0 = np.ones(instance.n)
        EQ_vec0[0::3] *= instance.EQX[0]  # 每隔3个，自由度0, 3, 6, ... → X方向
        EQ_vec0[1::3] *= instance.EQY[0]  # 自由度1, 4, 7, ... → Y方向
        EQ_vec0[2::3] *= instance.EQZ[0]  # 自由度2, 5, 8, ... → Z方向
        instance.dt = instance.time[1] - instance.time[0]
        P0 = -instance.M @ EQ_vec0
        instance.P_m0 = instance.eigenvectors.T @ P0
        instance.q0 = np.array([(instance.eigenvectors[:, i].T @ instance.M @ instance.InitialU) / (
                    instance.eigenvectors[:, i].T @ instance.M @ instance.eigenvectors[:, i]) for i in
                                range(instance.n)])
        instance.qq0 = np.array([(instance.eigenvectors[:, i].T @ instance.M @ instance.InitialU_dot) / (
                    instance.eigenvectors[:, i].T @ instance.M @ instance.eigenvectors[:, i]) for i in
                                 range(instance.n)])

        instance.qqq0 = np.linalg.solve(instance.M_modal,
                                        instance.P_m0 - instance.C_modal @ instance.qq0 - instance.K_modal @ instance.q0)

        instance.K_m_hat = instance.K_modal + (1 / (instance.bta * instance.dt ** 2)) * instance.M_modal + (
                    instance.gam / (instance.bta * instance.dt)) * instance.C_modal
        instance.a = (1 / (instance.bta * instance.dt)) * instance.M_modal + (
                    instance.gam / instance.bta) * instance.C_modal
        instance.b = (1 / (2 * instance.bta)) * instance.M_modal + instance.dt * (
                    instance.gam / (2 * instance.bta) - 1) * instance.C_modal

        # 进行 Newmark 积分并返回结果
        q = np.zeros((instance.N, instance.n))
        q[0, :] = instance.q0
        U = np.zeros((instance.N, instance.n))

        qq = np.zeros((instance.N, instance.n))
        qq[0, :] = instance.qq0
        U_dt = np.zeros((instance.N, instance.n))

        qqq = np.zeros((instance.N, instance.n))
        qqq[0, :] = instance.qqq0
        U_dt2 = np.zeros((instance.N, instance.n))

        for i in range(instance.N - 1):
            EQ_vec = np.ones(instance.n)
            EQ_vec[0::3] *= instance.EQX[i]  # 每隔3个，自由度0, 3, 6, ... → X方向
            EQ_vec[1::3] *= instance.EQY[i]  # 自由度1, 4, 7, ... → Y方向
            EQ_vec[2::3] *= instance.EQZ[i]  # 自由度2, 5, 8, ... → Z方向
            P = -instance.M @ EQ_vec
            P_m = instance.eigenvectors.T @ P
            EQ_vec1 = np.ones(instance.n)
            EQ_vec1[0::3] *= instance.EQX[i+1]  # 每隔3个，自由度0, 3, 6, ... → X方向
            EQ_vec1[1::3] *= instance.EQY[i+1]  # 自由度1, 4, 7, ... → Y方向
            EQ_vec1[2::3] *= instance.EQZ[i+1]  # 自由度2, 5, 8, ... → Z方向
            P1 = -instance.M @ EQ_vec1
            P_m1 = instance.eigenvectors.T @ P1
            DP_m = P_m1 - P_m
            DP_m_hat = DP_m + instance.a @ instance.qq0 + instance.b @ instance.qqq0
            Dq = np.linalg.solve(instance.K_m_hat, DP_m_hat).T
            Dqq = (instance.gam / (instance.bta * instance.dt)) * Dq - (
                        instance.gam / instance.bta) * instance.qq0 + instance.dt * (
                              1 - (instance.gam / (2 * instance.bta))) * instance.qqq0
            Dqqq = (1 / (instance.bta * instance.dt ** 2)) * Dq - (1 / (instance.bta * instance.dt)) * instance.qq0 - (
                        1 / (2 * instance.bta)) * instance.qqq0
            q[i + 1, :] = q[i, :] + Dq
            qq[i + 1, :] = qq[i, :] + Dqq
            qqq[i + 1, :] = qqq[i, :] + Dqqq
            instance.qq0 = instance.qq0 + Dqq
            instance.qqq0 = instance.qqq0 + Dqqq
            U[i + 1, :] = (instance.eigenvectors @ q[i + 1, :].T).T
            U_dt[i + 1, :] = (instance.eigenvectors @ qq[i + 1, :].T).T
            U_dt2[i + 1, :] = (instance.eigenvectors @ qqq[i + 1, :].T).T

        Result = {'time': instance.time,'U': U, 'U_dt': U_dt, 'U_dt2': U_dt2, 'DOF': instance.n}
        return Result
