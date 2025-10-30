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
import matplotlib.pyplot as plt
import numpy as np
from ..config import DEVICE
class Visualizer:
    def __init__(self):
        pass  # 初始化方法不再需要接收 losses 参数

    def plot_loss_curve(self, losses):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)  # 创建一个子图
        plt.semilogy(losses, marker='o', linestyle='-')  # 添加损失曲线
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss Curve')
        plt.grid()
        # 保存损失值
        np.savetxt('losses.txt', losses, delimiter='\t', header='Loss', comments='')

    @staticmethod
    def plot_displacement(trainerresults, num_timepoints=2000):
        duration_time = trainerresults['duration_time']
        dimensionsless_Pt = trainerresults['dimensionsless_Pt']
        net = trainerresults['net']
        t_test = torch.linspace(0, duration_time, num_timepoints, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            u_pred = net(t_test) * dimensionsless_Pt

        plt.subplot(2, 1, 2)  # 创建第二个子图
        for i in range(u_pred.shape[1]):
            plt.plot(t_test.cpu().numpy(), u_pred.cpu().numpy()[:, i], label=f'DOF {i + 1}', linewidth=2)
        plt.legend()
        plt.xlabel('Time (t_test)')
        plt.ylabel('Predicted Displacement (u_pred)')
        plt.title('Predicted Displacement for All DOFs')
        plt.grid(True, linestyle=':', alpha=1)

        plt.tight_layout()  # 自动调整子图间距
        plt.ioff()
        plt.show()
        # 保存预测位移
        u_pred_np = u_pred.cpu().numpy()
        time_data = t_test.cpu().numpy()
        data_to_save = np.column_stack((time_data, u_pred_np))  # 合并时间和位移数据
        np.savetxt('predicted_displacements.txt', data_to_save, delimiter='\t',
                   header='Time\t' + '\t'.join([f'DOF {i + 1}' for i in range(u_pred_np.shape[1])]), comments='')
