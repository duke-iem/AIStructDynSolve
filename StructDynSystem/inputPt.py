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

class inputPt:
    def __init__(self, StructDynSystemParm, duration_time, pt_time, pt_value):
        self.duration_time = duration_time
        self.pt_time = pt_time
        self.pt_value = pt_value
        # 返回所有中值中的中值
        median_value = np.median([np.median(np.abs(Pt)) if isinstance(Pt, np.ndarray) else Pt.abs().median() for Pt in self.pt_value])
        self.dimensionsless_Pt = median_value / StructDynSystemParm.get_dimensionsless_input() /StructDynSystemParm.get_dimensionsless_unit()

    def get_Ptparm(self):
        return self.duration_time, self.pt_time, self.pt_value, self.dimensionsless_Pt

    def get_duration_time(self):
        return self.duration_time

    def get_dimensionsless_Pt(self):
        return self.dimensionsless_Pt