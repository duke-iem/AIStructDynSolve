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

class inputPtFunc:
    def __init__(self, StructDynSystemParm, duration_time, pt_Func):
        self.duration_time = duration_time
        self.pt_Func = pt_Func
        # # 求Pt_func的最大值
        time_steps = torch.arange(0, duration_time, 0.01)  # 从0到duration_time，间隔dt
        all_median_values = []
        # 对每个时间点t，获取Pt_func(t)，然后找到每个张量的最大值
        for t in time_steps:
            Pt_list = pt_Func(t)  # 获取 Pt_func 返回的张量列表
            # 对列表中的每个张量求均值，并存储
            for Pt in Pt_list:
                all_median_values.append(Pt.abs().median().item())  # 求该张量的均值，并存储
        # 返回所有均值中的均值
        median_value = np.median(all_median_values)
        self.dimensionsless_Pt = median_value / StructDynSystemParm.get_dimensionsless_input() /StructDynSystemParm.get_dimensionsless_unit()

    def get_Ptparm(self):
        return self.duration_time, self.pt_Func, self.dimensionsless_Pt

    def get_duration_time(self):
        return self.duration_time

    def get_dimensionsless_Pt(self):
        return self.dimensionsless_Pt