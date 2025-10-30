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
import os

class Predictor:
    def __init__(self, model_path, num_timepoints=2000):
        self.model_path = model_path
        self.num_timepoints = num_timepoints
        self.model_loaded, self.duration_time, self.dimensionsless_Pt = self.load_model()

    def load_model(self):
        # 确保模型路径有效
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f'Model file {self.model_path} not found.')
        # 加载模型数据
        modeldata = torch.load(self.model_path, map_location=DEVICE,weights_only=False)
        # 获取模型和其他参数
        duration_time = modeldata['duration_time']
        dimensionsless_Pt = modeldata['dimensionsless_Pt']
        # 加载模型并设置为评估模式
        model = modeldata['net']
        model.eval()  # 设置模型为evaluation状态
        return model, duration_time, dimensionsless_Pt

    def predict_displacements(self):
        t_test = torch.linspace(0, self.duration_time, self.num_timepoints, device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            u_pred = self.model_loaded(t_test)* self.dimensionsless_Pt
        return t_test, u_pred

    def gradientscompute(self, u, x, order=1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u, device=DEVICE),
                                       create_graph=True, only_inputs=True)[0]
        else:
            return self.gradientscompute(self.gradientscompute(u, x), x, order=order - 1)

    def predict_velocities(self):
        t_test = torch.linspace(0, self.duration_time, self.num_timepoints, device=DEVICE).unsqueeze(1)
        t_test.requires_grad_(True)  # 允许对 t_test 计算梯度
        u_pred = self.model_loaded(t_test)
        # 计算速度
        u_pred_dot = []
        for i in range(u_pred.shape[1]):
            u_pred_dot.append(self.gradientscompute(u_pred[:, i], t_test, 1))
        u_pred_dot = torch.cat(u_pred_dot, dim=1).to(DEVICE)* self.dimensionsless_Pt

        return t_test, u_pred_dot

    def predict_accelerations(self):
        t_test = torch.linspace(0, self.duration_time, self.num_timepoints, device=DEVICE).unsqueeze(1)
        t_test.requires_grad_(True)  # 允许对 t_test 计算梯度
        u_pred = self.model_loaded(t_test)
        # 计算加速度
        u_pred_dotdot = []

        for i in range(u_pred.shape[1]):
            u_pred_dotdot.append(self.gradientscompute(u_pred[:, i], t_test, 2))
        u_pred_dotdot = torch.cat(u_pred_dotdot, dim=1).to(DEVICE)* self.dimensionsless_Pt

        return t_test, u_pred_dotdot

    @staticmethod
    def plot_displacements(t_test, u_pred):
        for i in range(u_pred.shape[1]):
            plt.plot(t_test.cpu().numpy(), u_pred[:, i].cpu().numpy(), label=f'DOF {i + 1}', linewidth=2)
        plt.legend()
        plt.xlabel('Time (t_test)')
        plt.ylabel('Predicted Displacements (u_pred)')
        plt.title('Predicted Displacements for All DOFs')
        plt.grid(True, linestyle=':', alpha=1)
        plt.show()

    @staticmethod
    def plot_velocities(t_test, velocities):
        for i in range(velocities.shape[1]):
            plt.plot(t_test.detach().cpu().numpy(), velocities[:, i].detach().cpu().numpy(), label=f'DOF {i + 1}', linewidth=2)
        plt.legend()
        plt.xlabel('Time (t_test)')
        plt.ylabel('Predicted Velocities (vel_pred)')
        plt.title('Predicted Velocities for All DOFs')
        plt.grid(True, linestyle=':', alpha=1)
        plt.show()

    @staticmethod
    def plot_accelerations(t_test, accelerations):
        for i in range(accelerations.shape[1]):
            plt.plot(t_test.detach().cpu().numpy(), accelerations[:, i].detach().cpu().numpy(), label=f'DOF {i + 1}', linewidth=2)
        plt.legend()
        plt.xlabel('Time (t_test)')
        plt.ylabel('Predicted accelerations (acc_pred)')
        plt.title('Predicted accelerations for All DOFs')
        plt.grid(True, linestyle=':', alpha=1)
        plt.show()

    @staticmethod
    def save_displacements(t_test, u_pred, filename='predicted_displacements.txt'):
        u_pred_np = u_pred.cpu().numpy()
        time_data = t_test.cpu().numpy()
        data_to_save = np.column_stack((time_data, u_pred_np))  # 合并时间和位移数据
        np.savetxt(filename, data_to_save, delimiter='\t',
                   header='Time\t' + '\t'.join([f'DOF {i + 1}' for i in range(u_pred_np.shape[1])]), comments='')

    @staticmethod
    def save_velocities(t_test, velocities, filename='predicted_velocities.txt'):
        velocities_np = velocities.detach().cpu().numpy()
        time_data = t_test.detach().cpu().numpy()
        data_to_save = np.column_stack((time_data, velocities_np))  # 合并时间和速度数据
        np.savetxt(filename, data_to_save, delimiter='\t',
                   header='Time\t' + '\t'.join([f'DOF {i + 1}' for i in range(velocities_np.shape[1])]), comments='')

    @staticmethod
    def save_accelerations(t_test, accelerations, filename='predicted_accelerations.txt'):
        accelerations_np = accelerations.detach().cpu().numpy()
        time_data = t_test.detach().cpu().numpy()
        data_to_save = np.column_stack((time_data, accelerations_np))  # 合并时间和加速度数据
        np.savetxt(filename, data_to_save, delimiter='\t',
                   header='Time\t' + '\t'.join([f'DOF {i + 1}' for i in range(accelerations_np.shape[1])]), comments='')


    def displacements(self):
        t_test, u_pred = self.predict_displacements()
        self.plot_displacements(t_test, u_pred)
        self.save_displacements(t_test, u_pred)

    def velocities(self):
        t_test_vel, velocities = self.predict_velocities()
        self.plot_velocities(t_test_vel, velocities)
        self.save_velocities(t_test_vel, velocities)


    def accelerations(self):
        t_test_accel, accelerations = self.predict_accelerations()
        self.plot_accelerations(t_test_accel, accelerations)
        self.save_accelerations(t_test_accel, accelerations)
