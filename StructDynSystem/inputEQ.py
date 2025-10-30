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

class inputEQ:
    def __init__(self, StructDynSystemParm, duration_time, eq_time, eq_acc):
        self.duration_time = duration_time
        self.eq_time = eq_time
        self.eq_acc = eq_acc
        if not isinstance(self.eq_acc, np.ndarray):
            # 如果self.eq_acc不是np.array矩阵
            self.dimensionsless_Pt = self.eq_acc.abs().median() / StructDynSystemParm.get_dimensionsless_input()
        else:
            # 如果是np.array矩阵
            self.dimensionsless_Pt = np.median(np.abs(self.eq_acc)) / StructDynSystemParm.get_dimensionsless_input()

    def get_EQparm(self):
        return self.duration_time, self.eq_time, self.eq_acc, self.dimensionsless_Pt

    def get_duration_time(self):
        return self.duration_time

    def get_dimensionsless_Pt(self):
        return self.dimensionsless_Pt
