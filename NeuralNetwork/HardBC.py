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

"""Hard Boundary Conditions硬边界约束y(0) = 0和y'(0) = 0"""
import torch
from ..config import DEVICE

class HardBC:
    def __init__(self, net):
        self.net = net
        self.DOF = self.get_dof()  # 动态获取自由度的数量
        self.net.apply_output_transform(self.hard_BC)

    def get_dof(self):
        # 使用小的测试输入获取 DOF
        test_input = torch.zeros(1, 1)  # 示例输入，形状为 [1, 1]
        output = self.net(test_input)
        return output.shape[1]  # 获取输出的列数作为自由度

    def hard_BC(self, x, f):
        # 扩展 x 到所有自由度，形状为 [batch_size, DOF]
        x_AllDOF = torch.concat([x] * self.DOF, dim=1).to(DEVICE)
        #U = f * x_AllDOF * x_AllDOF
        #U = f * (x_AllDOF * torch.log(x_AllDOF + 1))
        #U = f * (x_AllDOF - torch.log(x_AllDOF + 1))
        #U = f * (torch.tanh(x_AllDOF) - torch.log(x_AllDOF + 1))
        U = f * torch.tanh(x_AllDOF) * torch.tanh(x_AllDOF)
        return U

