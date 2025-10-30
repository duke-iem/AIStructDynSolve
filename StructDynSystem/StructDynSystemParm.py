"""
AIStructDynSolve
A framework focused on solving structural dynamics problems using artificial intelligence (AI) methods
solve the following ODE of MDOF
M*U_dotdot+C*U_dot+K*U=Pt
initial condition: U(t=0)=InitialU, and U_dot(t=0)=InitialU_dot
Author: 杜轲 duke@iem.ac.cn
Date: 2023/12/26
"""
import torch
import numpy as np
from ..config import DEVICE, DEFAULT_DTYPE

#定义结构动力学参数的类
class StructDynSystemParm:
    def __init__(self, mass_values, damping_matrix, stiffness_matrix):
        self.mass_values = mass_values
        self.damping_matrix = damping_matrix
        self.stiffness_matrix = stiffness_matrix

        # 假设 mass_values, damping_matrix 和 stiffness_matrix 可能是 NumPy 数组或张量
        if isinstance(mass_values, torch.Tensor):
            self.M0_tensor = mass_values.to(DEFAULT_DTYPE).to(DEVICE)
        else:
            self.M0_tensor = torch.from_numpy(mass_values).to(DEFAULT_DTYPE).to(DEVICE)

        if isinstance(damping_matrix, torch.Tensor):
            self.C0_tensor = damping_matrix.to(DEFAULT_DTYPE).to(DEVICE)
        else:
            self.C0_tensor = torch.from_numpy(damping_matrix).to(DEFAULT_DTYPE).to(DEVICE)

        if isinstance(stiffness_matrix, torch.Tensor):
            self.K0_tensor = stiffness_matrix.to(DEFAULT_DTYPE).to(DEVICE)
        else:
            self.K0_tensor = torch.from_numpy(stiffness_matrix).to(DEFAULT_DTYPE).to(DEVICE)

        self.M, self.C, self.K, self.dimensionsless_unit = self.non_dimensionalize()

    def non_dimensionalize(self):
        dimensionsless_unit = torch.min(torch.diag(self.M0_tensor)).to(DEVICE)
        M_tensor = self.M0_tensor / dimensionsless_unit
        K_tensor = self.K0_tensor / dimensionsless_unit
        C_tensor = self.C0_tensor / dimensionsless_unit
        return M_tensor, C_tensor, K_tensor, dimensionsless_unit

    def get_matrices(self):
        return self.M, self.C, self.K

    def get_dimensionsless_unit(self):
        return self.dimensionsless_unit

    def get_dimensionsless_input(self):
        #value = (torch.max(self.K0_tensor) / torch.max(self.M0_tensor) / 10.0).item()
        value = 20.0
        return value