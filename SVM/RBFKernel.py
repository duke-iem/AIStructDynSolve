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

class RBFKernel:
    @staticmethod
    def RBF_Kernel_Matrix(t_j, sigma2, t_k=None):
        """
        计算径向基函数 (RBF) 核矩阵。

        参数:
        - t_j (array): 时间向量 j。
        - sigma2 (float): RBF 的方差参数。
        - t_k (array, 可选): 时间向量 k。如果为 None，则使用 t_j。

        返回:
        - K ( array): RBF 核矩阵。
        """
        # 转换为列向量
        t_j = np.atleast_2d(t_j).T if t_j.ndim == 1 else t_j

        if t_k is None:
            # 创建对称的 NxN 核矩阵
            Tj, Tk = np.meshgrid(t_j, t_j)
        else:
            # 转换为列向量
            t_k = np.atleast_2d(t_k).T if t_k.ndim == 1 else t_k
            # 创建 N_tk x N_j 核矩阵
            Tj, Tk = np.meshgrid(t_j, t_k)

        # 计算差异矩阵
        t_mat_diff = Tk - Tj
        # 计算 RBF 核
        K = np.exp(-t_mat_diff ** 2 / (2 * sigma2))
        return K

    @staticmethod
    def RBF_Kernel_Derivatives(tj, tk, sigma2):
        """
        计算 RBF 核矩阵的导数。

        参数:
        - tj (array): 时间向量 j。
        - tk (array): 时间向量 k。
        - sigma2 (float): RBF 的方差参数。

        返回:
        - result (dict): 包含导数矩阵的字典。
        """
        Ke = RBFKernel.RBF_Kernel_Matrix(tj, sigma2, tk)
        tj_mat = np.tile(tj, (len(tk), 1))
        tk_mat = np.tile(tk.reshape(-1, 1), (1, len(tj)))

        C10 = -(1 / sigma2) * (tj_mat - tk_mat)
        C11 = -(1 / sigma2 ** 2) * (tj_mat - tk_mat) ** 2 + (1 / sigma2)
        C21 = (1 / sigma2 ** 3) * (tj_mat - tk_mat) ** 3 - (3 / sigma2 ** 2) * (tj_mat - tk_mat)
        C22 = (1 / sigma2 ** 4) * (tj_mat - tk_mat) ** 4 - (6 / sigma2 ** 3) * (tj_mat - tk_mat) ** 2 + (3 / sigma2 ** 2)

        K10 = Ke * C10
        K01 = -K10
        K11 = Ke * C11
        K20 = -K11
        K02 = K20
        K21 = Ke * C21
        K12 = -K21
        K22 = Ke * C22

        result = {
            'K00': Ke,
            'K01': K01,
            'K02': K02,
            'K10': K10,
            'K11': K11,
            'K12': K12,
            'K20': K20,
            'K21': K21,
            'K22': K22
        }
        return result
