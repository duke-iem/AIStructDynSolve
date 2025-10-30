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
from ..config import DEVICE
class Gradients:
    @staticmethod
    def compute(u, x, order=1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u, device=DEVICE),
                                       create_graph=True, only_inputs=True)[0]
        else:
            return Gradients.compute(Gradients.compute(u, x), x, order=order - 1)
