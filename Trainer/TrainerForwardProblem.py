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
import time
from torch.optim import LBFGS  # L-BFGS优化算法
from ..config import DEVICE

class TrainerForwardProblem:
    def __init__(self, net, loss_functions, Adamsteps, LBFGSsteps, loss_weight=None, HardBC=False):
        if HardBC:
            # 启用硬边界条件时，不考虑损失项
            self.lossweight = None
        else:
            if loss_weight is None:
                loss_weight = [1.0, 1.0, 1.0]  # 默认使用三个损失权重
            self.lossweight = loss_weight

        self.HardBC = HardBC
        self.net = net
        self.loss_functions = loss_functions
        self.Adamsteps = Adamsteps
        self.LBFGSsteps = LBFGSsteps
        self.duration_time = loss_functions.gettime()
        self.dimensionsless_Pt = loss_functions.get_dimensionsless_Pt()
        self.total_training_time = 0  # 记录总训练时间
        self.losses = []
        self.results = {}

    def train_with_adam(self, init_lr=0.001, scheduler_type='StepLR', step_size=1000, gamma=1.0,T_max=200, factor=0.1, patience=10, show_step=100):
        opt = torch.optim.Adam(params=self.net.parameters(), lr=init_lr)
        # 选择学习率调度器
        if scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=factor, patience=patience)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        # 使用Adam训练
        print("Start training......")
        print(f"Start Adam training for {self.Adamsteps} steps")
        if self.HardBC:
            print("Step    ", "   Training loss(lossODE)       ", "Training total loss")
        else:
            print("Step    ", "   Training loss(lossODE,lossInitialU,lossInitialU_dot)       ", "Training total loss")
        # 开始计时
        start_time = time.time()
        for i in range(self.Adamsteps):
            opt.zero_grad()
            if self.HardBC:
                totalloss = self.loss_functions.lossODE()
            else:
                totalloss = self.lossweight[0] * self.loss_functions.lossODE() + \
                            self.lossweight[1] * self.loss_functions.lossInitialU() + \
                            self.lossweight[2] * self.loss_functions.lossInitialU_dot()
            plt.ion()  # 开启交互模式
            if i % show_step == 0:
                if self.HardBC:
                    print(i, "   ", [self.loss_functions.lossODE().item()], "   ",[totalloss.item()])
                else:
                    print(i, "   ", [self.loss_functions.lossODE().item(),
                                             self.loss_functions.lossInitialU().item(),
                                             self.loss_functions.lossInitialU_dot().item()], "   ",
                          [totalloss.item()])
                t_test = torch.linspace(0, self.duration_time, 2000, device=DEVICE).unsqueeze(1)  # 设置测试时间
                with torch.no_grad():
                    u_pred = self.net(t_test) * self.dimensionsless_Pt  # 计算预测位移
                    plt.cla()
                    for j in range(u_pred.shape[1]):  # 遍历每组数据
                        plt.plot(t_test.cpu().numpy(), u_pred.detach().cpu()[:, j], label=f'DOF {j + 1}', linewidth=3)
                    plt.legend()
                    plt.xlabel('t_test')
                    plt.ylabel('u_pred')
                    plt.title('Predicted Displacement for All DOFs')
                    plt.show()
                    plt.pause(0.1)
            totalloss.backward()
            opt.step()
            scheduler.step()
            self.losses.append(totalloss.item())

        end_time = time.time()
        adam_training_time = end_time - start_time
        self.total_training_time += adam_training_time  # 更新总训练时间
    def train_with_lbfgs(self, max_iter=1000, tolerance_grad=0, tolerance_change=0, line_search_fn='strong_wolfe',show_step=100):
        lbfgs_opt = LBFGS(self.net.parameters(), max_iter=max_iter, tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, line_search_fn=line_search_fn)
        lbfgsiteration = 0
        start_time = time.time()

        def closure():
            nonlocal lbfgsiteration
            lbfgs_opt.zero_grad()
            if self.HardBC:
                totalloss = self.loss_functions.lossODE()
            else:
                totalloss = self.lossweight[0] * self.loss_functions.lossODE() + \
                            self.lossweight[1] * self.loss_functions.lossInitialU() + \
                            self.lossweight[2] * self.loss_functions.lossInitialU_dot()
            if lbfgsiteration % show_step == 0:
                if self.HardBC:
                    print(lbfgsiteration, "   ", [self.loss_functions.lossODE().item()], "   ",[totalloss.item()])
                else:
                    print(lbfgsiteration, "   ", [self.loss_functions.lossODE().item(),
                                             self.loss_functions.lossInitialU().item(),
                                             self.loss_functions.lossInitialU_dot().item()], "   ",
                          [totalloss.item()])
                t_test = torch.linspace(0, self.duration_time, 2000, device=DEVICE).unsqueeze(1)
                with torch.no_grad():
                    u_pred = self.net(t_test) * self.dimensionsless_Pt
                    plt.cla()
                    for i in range(u_pred.shape[1]):
                        plt.plot(t_test.cpu().numpy(), u_pred.cpu().numpy()[:, i], label=f'DOF {i + 1}', linewidth=3)
                    plt.legend()
                    plt.xlabel('t_test')
                    plt.ylabel('u_pred')
                    plt.title('Predicted Displacement for All DOFs')
                    plt.show()
                    plt.pause(0.1)
            totalloss.backward()
            self.losses.append(totalloss.item())

            lbfgsiteration += 1  # 更新迭代次数

            if lbfgsiteration >= self.LBFGSsteps:  # 检查是否达到最大迭代次数
                raise StopIteration  # 抛出异常以停止优化

            return totalloss

        print(f"Start L-BFGS training for {self.LBFGSsteps} steps")
        try:
            while True:
                lbfgs_opt.step(closure)
        except StopIteration:
            print("Training finished")
            # 结束计时
            end_time = time.time()
            lbfgs_training_time = end_time - start_time
            self.total_training_time += lbfgs_training_time  # 更新总训练时间
            print(f"Total training time: {self.total_training_time} seconds")
            # 保存训练结果，将模型和标量存储在一个字典中
            self.results = {
                'net': self.net,
                'dimensionsless_Pt': self.dimensionsless_Pt,
                'duration_time': self.duration_time,
            }

