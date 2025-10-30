"""
AIStructDynSolve
A framework focused on solving structural dynamics problems using artificial intelligence (AI) methods
solve the following ODE of MDOF
M*U_dotdot+C*U_dot+K*U=Pt
initial condition: U(t=0)=InitialU, and U_dot(t=0)=InitialU_dot
Author: 杜轲 duke@iem.ac.cn
Date: 2025/03/18
"""
import torch
import numpy as np

class inputEQ3D:
    def __init__(self, StructDynSystemParm, duration_time, eq_time, eq_accX, eq_accY, eq_accZ):
        self.duration_time = duration_time
        self.eq_time = eq_time
        self.eq_accX = eq_accX
        self.eq_accY = eq_accY
        self.eq_accZ = eq_accZ
        if not isinstance(self.eq_accX, np.ndarray):
            # 如果self.eq_acc不是np.array矩阵
            self.dimensionsless_Pt = self.eq_accX.abs().median() / StructDynSystemParm.get_dimensionsless_input()
        else:
            # 如果是np.array矩阵
            self.dimensionsless_Pt = np.median(np.abs(self.eq_accX)) / StructDynSystemParm.get_dimensionsless_input()

    def get_EQparm(self):
        return self.duration_time, self.eq_time, self.eq_accX, self.eq_accY, self.eq_accZ, self.dimensionsless_Pt

    def get_duration_time(self):
        return self.duration_time

    def get_dimensionsless_Pt(self):
        return self.dimensionsless_Pt
