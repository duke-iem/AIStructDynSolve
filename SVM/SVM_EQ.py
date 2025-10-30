"""
AIStructDynSolve
A framework focused on solving structural dynamics problems using artificial intelligence (AI) methods
solve the following ODE of MDOF
M*U_dotdot+C*U_dot+K*U=Pt
initial condition: U(t=0)=InitialU, and U_dot(t=0)=InitialU_dot
Author: 杜轲 duke@iem.ac.cn
Date: 2024/11/16
"""

import numpy as np
from .RBFKernel import RBFKernel
import scipy

class SVM_EQ:
    def __new__(cls, M, C, K, t, EQ, gamma, sigma2, InitialU=None, InitialU_dot=None, PHI=None):
        """
        使用 __new__ 方法进行初始化，并直接返回计算结果。

        参数:
        - M (2D array): 质量矩阵。
        - C (2D array): 阻尼矩阵。
        - K (2D array): 刚度矩阵。
        - gamma (float): 优化参数 gamma。
        - sigma2 (float): 优化参数 sigma 平方。
        - t (array): 时间向量。
        - EQ (2D array): 外部载荷矩阵。
        - U_t0 (array): 初始位移。
        - U_dt0 (array): 初始速度。
        - PHI (2D array, 可选): 模态形状矩阵。如果提供，则转换到模态坐标。

        返回:
        - Result (dict): 包含位移、速度和加速度的字典。
        """
        # 调用父类的 __new__ 方法创建实例
        instance = super(SVM_EQ, cls).__new__(cls)

        # 初始化

        if isinstance(M, np.ndarray):
            dimensionsless_unit = np.max(np.diag(M))
            instance.M = M /dimensionsless_unit
        else:
            dimensionsless_unit = np.max(np.diag(M.numpy()))
            instance.M = M.numpy() /dimensionsless_unit

        if isinstance(C, np.ndarray):
            instance.C = C /dimensionsless_unit
        else:
            instance.C = C.numpy() /dimensionsless_unit

        if isinstance(K, np.ndarray):
            instance.K = K /dimensionsless_unit
        else:
            instance.K = K.numpy() /dimensionsless_unit

        instance.gamma = gamma
        instance.sigma2 = sigma2
        if not isinstance(EQ, np.ndarray):
            # 如果EQ不是np.array矩阵
            GM = np.array(EQ)
        else:
            # 如果EQ已经是np.array矩阵，直接赋值
            GM = EQ
        DOF = instance.M.shape[0]
        GM = GM.reshape((-1, 1))  # 调整以适应任意长度
        ones_matrix = np.ones((1, DOF))
        product = -instance.M @ (GM * ones_matrix).T
        P = product.T
        # 如果没有提供 InitialU 和 InitialU_dot，初始化为 0.0
        instance.InitialU = np.zeros(instance.M.shape[0]) if InitialU is None else InitialU
        instance.InitialU_dot = np.zeros(instance.M.shape[0]) if InitialU_dot is None else InitialU_dot
        # 调用 Implement_SVMEQ 计算并直接返回结果
        return instance.Implement_SVMEQ(instance.M, instance.C, instance.K, t, P, instance.InitialU, instance.InitialU_dot, PHI)

    def SVMEQ_method(self, M, C, K, time, EQ, U_t0, U_dt0):
        """
        实现 SVMEQ 算法进行动态结构分析。

        参数:
        - t (array): 时间向量。
        - EQ (2D array): 外部载荷矩阵。
        - U_t0 (array): 初始位移。
        - U_dt0 (array): 初始速度。

        返回:
        - Result (dict): 包含位移、速度和加速度的字典。
        """
        gamma, sigma2 = self.gamma, self.sigma2
        n = M.shape[0]
        if n != C.shape[0] or n != K.shape[0]:
            raise ValueError('输入信息有误，请重新输入正确信息')
        t = np.array(time)
        P = EQ
        if len(t) != P.shape[0]:
            raise ValueError('输入的动态载荷有误，请检查')
        t0 = t[0]
        t0 = np.array([t0])
        p_t0 = P[0, :]
        t_j = t[1:]
        P_tj = P[1:, :]
        N = len(t_j)

        # 计算 RBF 核导数
        K_Deri = RBFKernel.RBF_Kernel_Derivatives(t_j, t_j, sigma2)
        K00, K01, K02 = K_Deri['K00'], K_Deri['K01'], K_Deri['K02']
        K10, K11, K12 = K_Deri['K10'], K_Deri['K11'], K_Deri['K12']
        K20, K21, K22 = K_Deri['K20'], K_Deri['K21'], K_Deri['K22']

        # 计算初始时间 t0 的 RBF 核导数
        K_Deri_t0 = RBFKernel.RBF_Kernel_Derivatives(t0, t_j, sigma2)
        K02_t0, K01_t0, K00_t0 = K_Deri_t0['K02'], K_Deri_t0['K01'], K_Deri_t0['K00']
        K12_t0, K11_t0, K10_t0 = K_Deri_t0['K12'], K_Deri_t0['K11'], K_Deri_t0['K10']

        # 计算 t0 和 t0 的 RBF 核导数
        K_Deri_t0_s = RBFKernel.RBF_Kernel_Derivatives(t0, t0, sigma2)
        K00_t0_s = K_Deri_t0_s['K00']
        K10_t0_s = K_Deri_t0_s['K10']
        K01_t0_s = K_Deri_t0_s['K01']
        K11_t0_s = K_Deri_t0_s['K11']

        # 构建系数矩阵
        Kappa = np.zeros((n, n * N))
        H = np.zeros((n * N, n * N))
        B = np.zeros((n * N, n))
        T = np.zeros((n * N, n))
        Peff = np.zeros((n * N, 1))

        for i in range(n):
            K_int = np.outer(np.ones(N), K[i, :])
            Kappa[:, N * i:N * (i + 1)] = K_int.T
            for v in range(n):
                mi_Av_K2 = np.dot(M[i, :], M[v, :].T) * K22 + np.dot(M[i, :], C[v, :].T) * K12 + np.dot(M[i, :], K[v, :].T) * K02
                ci_Av_K1 = np.dot(C[i, :], M[v, :].T) * K21 + np.dot(C[i, :], C[v, :].T) * K11 + np.dot(C[i, :], K[v, :].T) * K01
                ki_Av_K0 = np.dot(K[i, :], M[v, :].T) * K20 + np.dot(K[i, :], C[v, :].T) * K10 + np.dot(K[i, :], K[v, :].T) * K00
                H[N * i:N * (i + 1), N * v:N * (v + 1)] = mi_Av_K2 + ci_Av_K1 + ki_Av_K0

                Bi_K0 = (M[i, v] * K02_t0 + C[i, v] * K01_t0 + K[i, v] * K00_t0).reshape(-1)
                B[N * i:N * (i + 1), v] = Bi_K0

                Ti_K0 = (M[i, v] * K12_t0 + C[i, v] * K11_t0 + K[i, v] * K10_t0).reshape(-1)
                T[N * i:N * (i + 1), v] = Ti_K0
            Peff[N * i:N * (i + 1)] = P_tj[:, i].reshape(-1, 1)

        # 构建系统矩阵 A 和向量 b
        A = np.vstack([
            np.hstack([H + (1 / gamma) * np.eye(n * N), B, T, Kappa.T]),
            np.hstack([B.T, K00_t0_s * np.eye(n), K10_t0_s * np.eye(n), np.eye(n)]),
            np.hstack([T.T, K01_t0_s * np.eye(n), K11_t0_s * np.eye(n), np.zeros((n, n))]),
            np.hstack([Kappa, np.eye(n), np.zeros((n, n)), np.zeros((n, n))])
        ])
        b = np.concatenate([Peff.ravel(), U_t0, U_dt0, np.zeros(n)], axis=0)

        # 正则化以避免矩阵奇异
        lambda_identity = 1e-5 * np.eye(A.shape[0])
        A_regularized = A + lambda_identity

        # 求解线性方程组
        #result = np.linalg.solve(A_regularized, b)
        #result, *_ = scipy.linalg.lstsq(A_regularized, b)
        result = scipy.linalg.solve(A_regularized, b, assume_a='sym')
        Alpha = result[:n * N]
        Beta = result[n * N:n * (N + 1)]
        Tau = result[n * (N + 1):n * (N + 2)]
        bias = result[n * (N + 2):n * (N + 3)]

        # 初始化位移和速度矩阵
        U = np.zeros((N + 1, n))
        U_dt = np.zeros((N + 1, n))
        U_dt2 = np.zeros((N + 1, n))
        U[0, :] = U_t0
        U_dt[0, :] = U_dt0

        # 初始化 Omega 矩阵
        Omega = np.zeros((N, n * N))
        Omega_dt = np.zeros_like(Omega)
        Omega_dt2 = np.zeros_like(Omega)

        # 预测响应
        for i in range(n):
            for v in range(n):
                Omega[:, N * v:N * (v + 1)] = M[v, i] * K20 + C[v, i] * K10 + K[v, i] * K00
                Omega_dt[:, N * v:N * (v + 1)] = M[v, i] * K21 + C[v, i] * K11 + K[v, i] * K01
                Omega_dt2[:, N * v:N * (v + 1)] = M[v, i] * K22 + C[v, i] * K12 + K[v, i] * K02

            Ca = np.hstack((Omega, K00_t0, K10_t0, np.ones((N, 1))))
            Da = np.concatenate((Alpha, [Beta[i]], [Tau[i]], [bias[i]]))
            Ea = Ca @ Da
            U[1:, i] = Ea.flatten()

            Ca_dt = np.hstack((Omega_dt, K01_t0, K11_t0))
            Da_dt = np.concatenate((Alpha, [Beta[i]], [Tau[i]]))
            Ea_dt = Ca_dt @ Da_dt
            U_dt[1:, i] = Ea_dt.flatten()

            Ca_dt2 = np.hstack((Omega_dt2, K02_t0, K12_t0))
            Da_dt2 = np.concatenate((Alpha, [Beta[i]], [Tau[i]]))
            Ea_dt2 = Ca_dt2 @ Da_dt2
            U_dt2[1:, i] = Ea_dt2.flatten()

        # 创建结果字典
        Result = {'time': time,'U': U, 'U_dt': U_dt, 'U_dt2': U_dt2,'DOF': n}
        return Result

    def Implement_SVMEQ(self, M, C, K, t, EQ, U_t0, U_dt0, PHI=None):
        """
        实现 SVMEQ 方法，处理物理坐标和模态坐标。

        参数:
        - t (array): 时间向量。
        - EQ (2D array): 外部载荷矩阵。
        - U_t0 (array): 初始位移。
        - U_dt0 (array): 初始速度。
        - PHI (2D array, 可选): 模态形状矩阵。如果提供，则转换到模态坐标。

        返回:
        - Result (dict): 包含位移、速度和加速度的字典。
        """
        DOF = self.M.shape[0]
        if PHI is None:
            U_I = U_t0
            U_dtI = U_dt0
            Result_I = self.SVMEQ_method(M, C, K, t, EQ, U_I, U_dtI)
            U = Result_I['U']
            U_dt = Result_I['U_dt']
            U_dt2 = Result_I['U_dt2']
        else:
            # 转换到模态坐标
            M_modal = PHI.T @ self.M @ PHI
            C_modal = PHI.T @ self.C @ PHI
            K_modal = PHI.T @ self.K @ PHI
            P_eff = EQ @ PHI
            if np.linalg.norm(U_t0) + np.linalg.norm(U_dt0) == 0:
                U_I = U_t0
                U_dtI = U_dt0
            else:
                q0 = np.array([(PHI[:, i].T @ self.M @ U_t0) / (PHI[:, i].T @ self.M @ PHI[:, i]) for i in range(PHI.shape[1])])
                qq0 = np.array([(PHI[:, i].T @ self.M @ U_dt0) / (PHI[:, i].T @ self.M @ PHI[:, i]) for i in range(PHI.shape[1])])
                U_I = q0
                U_dtI = qq0

            # 实例化一个新的 SVMEQ_method 以使用模态矩阵
            Result_I = self.SVMEQ_method(M_modal, C_modal, K_modal,t, P_eff, U_I, U_dtI)
            U_modal = Result_I['U']
            U_dt_modal = Result_I['U_dt']
            U_dt2_modal = Result_I['U_dt2']

            # 转换回物理坐标
            U = U_modal @ PHI.T
            U_dt = U_dt_modal @ PHI.T
            U_dt2 = U_dt2_modal @ PHI.T

        Result = {'time': t,'U': U, 'U_dt': U_dt, 'U_dt2': U_dt2,'DOF': DOF}
        return Result
