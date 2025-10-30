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
from .Gradients import Gradients
from skopt.sampler import Lhs, Halton, Hammersly, Sobol
from ..config import DEVICE, DEFAULT_DTYPE

# 定义自定义函数组合
#def custom_func(t):
#    return np.sin(t)+np.cos(t)

class LossPtFunc:
    def __init__(
        self,
        net,
        struct_dyn_system,
        inputPt,
        number_ode,
        number_initial = None,  # Default is None
        observe_data = None,  # Default is None
        train_params = None,  # Default is None
        userdefined_train_params=None,  # Default is None
        userdefined_mass=None,
        userdefined_stiffness=None,
        userdefined_damping=None,
        loss_fn = 'L2Loss',  # Default to MSELoss, can be 'L1Loss', 'L2Loss'
        sampling_method='Hammersley',  # Default to Hammersley, can be 'Lhs', 'Halton', 'Sobol','Uniform'
        InitialU=0.0,  # Default is 0.0
        InitialU_dot=0.0,  # Default is 0.0
    ):
        self.net = net
        # 从 StructDynSystem 类实例获取 M, C, K 矩阵
        mass, damping, stiffness = struct_dyn_system.get_matrices()
        # Check if 'stiffness' "mass" "damping" is in trainable_params and set as a torch parameter if true
        self.dimensionsless_mass = mass
        self.dimensionsless_stiffness = stiffness
        self.dimensionsless_damping = damping

        if train_params is not None and "mass" in {key.lower() for key in train_params}:
            mask1 = mass != 0  # 创建一个布尔掩码，标记非零位置
            inv_mass = torch.zeros_like(mass)  # 初始化全 0 的矩阵
            inv_mass[mask1] = 1.0 / mass[mask1]  # 只对非零位置求倒数
            self.mass_dimensionsless = torch.nn.Parameter(mass * inv_mass , requires_grad=True)
            self.mass_mask = (mass != 0).to(mass)
        else:
            self.mass = mass

        if train_params is not None and "stiffness" in {key.lower() for key in train_params}:
            mask2 = stiffness != 0  # 创建一个布尔掩码，标记非零位置
            inv_stiffness = torch.zeros_like(stiffness)  # 初始化全 0 的矩阵
            inv_stiffness[mask2] = 1.0 / stiffness[mask2]  # 只对非零位置求倒数
            self.stiffness_dimensionsless = torch.nn.Parameter(stiffness * inv_stiffness, requires_grad=True)
            self.stiffness_mask = (stiffness != 0).to(stiffness)
        else:
            self.stiffness = stiffness

        if train_params is not None and "damping" in {key.lower() for key in train_params}:
            mask3 = damping != 0  # 创建一个布尔掩码，标记非零位置
            inv_damping = torch.zeros_like(damping)  # 初始化全 0 的矩阵
            inv_damping[mask3] = 1.0 / damping[mask3]  # 只对非零位置求倒数
            self.damping_dimensionsless = torch.nn.Parameter(damping * inv_damping, requires_grad=True)
            self.damping_mask = (damping != 0).to(damping)
        else:
            self.damping = damping

        self.dimensionsless_unit = struct_dyn_system.get_dimensionsless_unit()
        # 从 inputPt类获取 duration_time, pt_time, pt_value参数
        self.duration_time, self.Pt_Func,self.dimensionsless_Pt = inputPt.get_Ptparm()

        if userdefined_train_params is not None:
            for name in userdefined_train_params:
                value = userdefined_train_params[name]
                param = torch.nn.Parameter(value, requires_grad=True)
                setattr(self, name, param)

        self.DOF = mass.shape[0]  # 计算自由度
        self.number_ode = number_ode
        self.number_initial = number_initial
        self.observe_data = observe_data
        self.train_params = train_params
        self.userdefined_train_params = userdefined_train_params
        self.userdefined_mass = userdefined_mass
        self.userdefined_stiffness = userdefined_stiffness
        self.userdefined_damping = userdefined_damping
        self.sampling_method = sampling_method
        self.InitialU = InitialU
        self.InitialU_dot = InitialU_dot

        # Set the loss function, default is MSELoss
        loss_dict = {
            'L2Loss': torch.nn.MSELoss(reduction='mean'),
            'L1Loss': torch.nn.L1Loss(reduction='mean'),
        }
        self.loss_fn = loss_dict.get(loss_fn)
        self.loss_type = loss_fn  # 保留原始字符串类型用于判断
        if self.loss_fn is None:
            raise ValueError(f"Unsupported loss_fn type: {loss_fn}")

    def gettime(self):
        return self.duration_time

    def _get_sampler(self):
        """Returns the appropriate sampler based on the chosen sampling method."""
        samplers = {
            'Uniform': lambda: np.random.uniform(low=0.0, high=self.duration_time, size=(self.number_ode, 1)),
            'Lhs': Lhs,
            'Halton': Halton,
            'Sobol': Sobol,
            'Hammersley': Hammersly,
        }

        if self.sampling_method not in samplers:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}")

        sampler = samplers[self.sampling_method]
        if callable(sampler):
            return sampler()
        else:
            return sampler()

    def lossODE(self):
        # 在此处进行无量纲处理（缩放）
        scaled_params = {}
        if self.userdefined_train_params is not None:
            for name in self.userdefined_train_params:
                original_value = getattr(self, name)  # 动态访问 self.ea, self.ei 等
                scaled_params[name] = original_value
        if self.userdefined_mass is not None:
            self.mass = self.userdefined_mass(scaled_params)/self.dimensionsless_unit  # 执行传入的函数
        if self.userdefined_stiffness is not None:
            self.stiffness = self.userdefined_stiffness(scaled_params)/self.dimensionsless_unit # 执行传入的函数
        if self.userdefined_damping is not None:
            self.damping = self.userdefined_damping(scaled_params)/self.dimensionsless_unit  # 执行传入的函数
        # Get the appropriate sampler
        sampler = self._get_sampler()

        if self.sampling_method == 'Uniform':
            x_in = sampler  # For uniform, it's a direct numpy array
        else:
            # For other methods, generate samples and reshape them
            samples = sampler.generate(dimensions=[(0, self.duration_time)], n_samples=self.number_ode)
            x_in = np.array(samples).reshape(-1, 1)
        t_x_in = torch.from_numpy(x_in).to(DEFAULT_DTYPE).to(DEVICE).requires_grad_(True)
        u_collection = self.net(t_x_in)

        # Second and first derivatives for ODE calculation
        u_collection_dotdot = []
        u_collection_dot = []

        for i in range(self.DOF):
            u_collection_dotdot.append(Gradients.compute(u_collection[:, i], t_x_in, 2))
            u_collection_dot.append(Gradients.compute(u_collection[:, i], t_x_in, 1))

        u_collection_dotdot = torch.cat(u_collection_dotdot, dim=1).to(DEVICE)
        u_collection_dot = torch.cat(u_collection_dot, dim=1).to(DEVICE)

        #
        pt_list = self.Pt_Func(t_x_in)
        # 使用 torch.cat 进行列拼接
        Pt = torch.cat(pt_list, dim=1).to(DEVICE) / self.dimensionsless_unit /self.dimensionsless_Pt

        if self.train_params is not None and "mass" in {key.lower() for key in self.train_params}:
            m_masked = self.mass_dimensionsless * self.mass_mask
            self.mass = (m_masked + m_masked.T) / 2 * self.dimensionsless_mass
        if self.train_params is not None and "stiffness" in {key.lower() for key in self.train_params}:
            s_masked = self.stiffness_dimensionsless * self.stiffness_mask
            self.stiffness = (s_masked + s_masked.T) / 2 * self.dimensionsless_stiffness
        if self.train_params is not None and "damping" in {key.lower() for key in self.train_params}:
            d_masked = self.damping_dimensionsless * self.damping_mask
            self.damping = (d_masked + d_masked.T) / 2 * self.dimensionsless_damping
        # 计算整体的残差
        #residual = self.mass @ u_collection_dotdot.T + self.damping @ u_collection_dot.T + self.stiffness @ u_collection.T - Pt.T
        residual = torch.matmul(self.mass, u_collection_dotdot.T) + torch.matmul(self.damping, u_collection_dot.T) + torch.matmul(self.stiffness, u_collection.T) - Pt.T
        # 拆分为按行计算每个损失,使用torch.unbind()按行分解
        #residual_rows = torch.unbind(residual, dim=0)
        # 提取self.stiffness矩阵的对角元素
        #diagonal_values = torch.diag(self.stiffness)  # 获取对角线元素
        # 初始化总损失为0
        #total_loss = torch.tensor(0.0, device=DEVICE)
        # 对每一行计算损失并累加
        #for i, residual_row in enumerate(residual_rows):
        #    # 计算当前行的MSE损失
        #    row_loss = self.loss_fn(residual_row, torch.zeros_like(residual_row, device=DEVICE))
        #    # 将损失除以对应行的对角元素
        #    row_loss /= diagonal_values[i]
        #    # 累加到总损失
        #    total_loss += row_loss

        # 计算每行的MSE损失
        if self.loss_type == 'L2Loss':
            row_losses = torch.mean(residual ** 2, dim=1)
        elif self.loss_type == 'L1Loss':
            row_losses = torch.mean(torch.abs(residual), dim=1)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        # 提取self.stiffness矩阵的对角元素
        diagonal_values = torch.diag(self.dimensionsless_stiffness)  # 获取对角线元素
        # 归一化行损失
        normalized_row_losses = row_losses / diagonal_values
        # 累加总损失
        total_loss = torch.mean(normalized_row_losses)

        # Calculate ODE loss
        #u_collectionT = u_collection.T
        #total_loss = self.loss_fn(
        #    self.mass @ u_collection_dotdot.T + self.damping @ u_collection_dot.T + self.stiffness @ u_collection.T - Pt.T,
        #    torch.zeros_like(u_collectionT, device=DEVICE)
        #)

        return total_loss

    ## 求边界条件的损失函数
    def lossInitialU(self):
        if self.number_initial is None:
            raise ValueError(
                "number_initial must be specified when calling lossInitialU. Please provide number_initial during LossEQ initialization.")

        t_0 = torch.zeros(self.number_initial, 1, device=DEVICE).requires_grad_(True)  # 创建初始时间点
        u_0 = self.net(t_0)  # 神经网络输出
        loss_u = self.loss_fn(u_0.T, torch.zeros_like(u_0.T, device=DEVICE) + self.InitialU)
        return loss_u  # u(t=0)=0的损失

    def lossInitialU_dot(self):
        if self.number_initial is None:
            raise ValueError(
                "number_initial must be specified when calling lossInitialU_dot. Please provide number_initial during LossEQ initialization.")

        t_0 = torch.zeros(self.number_initial, 1, device=DEVICE).requires_grad_(True)  # 创建初始时间点
        u_0 = self.net(t_0)  # 神经网络输出

        # 计算u对x的一阶导数
        u_0_dot = [Gradients.compute(u_0[:, i], t_0, 1) for i in range(self.DOF)]
        u_0_dot = torch.cat(u_0_dot, dim=1).to(DEVICE)
        loss_u_dot = self.loss_fn(u_0_dot.T, torch.zeros_like(u_0.T, device=DEVICE) + self.InitialU_dot)

        return loss_u_dot  # u_dot(t=0)=0的损失

    ##求观测数据的损失
    def lossObserve(self):
        # 确保 observe_data 不为空
        if self.observe_data is not None:
            if not isinstance(self.observe_data[1], np.ndarray):
                # 如果self.observe_data[1]不是np.array矩阵
                observe_time = self.observe_data[1].to_numpy()  # 时间列
            else:
                observe_time = self.observe_data[1]  # 时间列
            observelist = self.observe_data[2]  # 位移或加速度观察值列表
            observelist_DOF = self.observe_data[3]  # 每列位移或加速度对应的自由度
            #observe_t_tensor = torch.tensor(observe_time.reshape(-1, 1), dtype=DEFAULT_DTYPE, device=DEVICE)
            total_loss = 0  # 初始化总损失

            # Get the appropriate sampler
            sampler = self._get_sampler()
            if self.sampling_method == 'Uniform':
                x_in = sampler  # For uniform, it's a direct numpy array
            else:
                # For other methods, generate samples and reshape them
                samples = sampler.generate(dimensions=[(0, self.duration_time)], n_samples=self.number_ode)
                x_in = np.array(samples).reshape(-1, 1)
            t_x_in = torch.from_numpy(x_in).to(DEFAULT_DTYPE).to(DEVICE).requires_grad_(True)

            # 判断是位移（Disp）还是加速度（Acc）
            if self.observe_data[0] == "Disp":
                # 对于位移的计算
                for dof, disp_observe in zip(observelist_DOF, observelist):
                    # 获取神经网络输出对应自由度的位移
                    disp_tensor = self.net(t_x_in)[:, dof - 1].unsqueeze(1)
                    # Interpolate external disp_observe
                    disp_observe_interpolation = np.interp(x_in, observe_time, disp_observe)
                    disp_observe_tensor = torch.from_numpy(disp_observe_interpolation).to(DEFAULT_DTYPE).to(DEVICE)/ self.dimensionsless_Pt
                    # 归一化（按最大值），避免除0
                    max_val = torch.max(torch.abs(disp_observe_tensor)).clamp(min=1e-8)
                    disp_observe_tensor_normalized = disp_observe_tensor / max_val
                    disp_tensor_normalized = disp_tensor / max_val
                    #disp_tensor = self.net(observe_t_tensor)[:, dof - 1].unsqueeze(1)
                    #if not isinstance(disp_observe, np.ndarray):
                    #    # 如果disp_observe不是np.array矩阵
                    #    disp_observe_tensor = torch.tensor(disp_observe.to_numpy().reshape(-1, 1), dtype=DEFAULT_DTYPE, device=DEVICE)/self.dimensionsless_Pt
                    #else:
                    #    disp_observe_tensor = torch.tensor(disp_observe.reshape(-1, 1), dtype=DEFAULT_DTYPE, device=DEVICE)/self.dimensionsless_Pt

                    # 计算归一化损失
                    loss_disp = self.loss_fn(disp_tensor_normalized, disp_observe_tensor_normalized)
                    # 累加损失
                    total_loss += loss_disp

            elif self.observe_data[0] == "Acc":
                # 对于加速度的计算
                for dof, acc_observe in zip(observelist_DOF, observelist):
                    # 计算获取神经网络输出对应自由度的加速度，这里假设 Gradients.compute 已经定义
                    acc_tensor = Gradients.compute(self.net(t_x_in)[:, dof - 1], t_x_in, 2)  # 二阶导数得到加速度
                    # Interpolate external acc_observe
                    acc_observe_interpolation = np.interp(x_in, observe_time, acc_observe)
                    acc_observe_tensor = torch.from_numpy(acc_observe_interpolation).to(DEFAULT_DTYPE).to(DEVICE)/ self.dimensionsless_Pt
                    # 归一化（按最大值），避免除0
                    max_val = torch.max(torch.abs(acc_observe_tensor)).clamp(min=1e-8)
                    acc_observe_tensor_normalized = acc_observe_tensor / max_val
                    acc_tensor_normalized = acc_tensor / max_val
                    #acc_tensor = Gradients.compute(self.net(observe_t_tensor)[:, dof - 1], observe_t_tensor, 2)  # 二阶导数得到加速度
                    #if not isinstance(acc_observe, np.ndarray):
                    #    # 如果acc_observe不是np.array矩阵
                    #    acc_observe_tensor = torch.tensor(acc_observe.to_numpy().reshape(-1, 1), dtype=DEFAULT_DTYPE, device=DEVICE)/self.dimensionsless_Pt
                    #else:
                    #    acc_observe_tensor = torch.tensor(acc_observe.reshape(-1, 1), dtype=DEFAULT_DTYPE, device=DEVICE)/self.dimensionsless_Pt

                    # 计算归一化损失
                    loss_acc = self.loss_fn(acc_tensor_normalized, acc_observe_tensor_normalized)
                    # 累加损失
                    total_loss += loss_acc

            else:
                print("Observe data must be Disp or Acc")
                return None  # 如果观测数据既不是位移也不是加速度，则返回 None

        else:
            print("No observe_data provided.")
            return None  # 如果没有提供 observe_data，则返回 None

        # 返回总损失
        return total_loss


    def get_dimensionsless_unit(self):
        return self.dimensionsless_unit

    def get_dimensionsless_mass(self):
        return self.dimensionsless_mass

    def get_dimensionsless_stiffness(self):
        return self.dimensionsless_stiffness

    def get_dimensionsless_damping(self):
        return self.dimensionsless_damping

    def get_dimensionsless_Pt(self):
        return self.dimensionsless_Pt

    def get_trainable_params_name(self):
        # 检查 self.train_params 是否为 None
        if self.train_params is None and self.userdefined_train_params is None:
            raise ValueError("train_params and userdefined_train_params are both None. Please provide at least one.")
        names = []
        if self.train_params is not None:
            names.extend(self.train_params)
        if self.userdefined_train_params is not None:
            names.extend(self.userdefined_train_params.keys())
        return names

    def get_trainable_params(self):
        # 检查 self.train_params 是否为 None
        if self.train_params is None and self.userdefined_train_params is None:
            raise ValueError("train_params is None. Please provide valid train_params.")
        # 初始化返回的参数列表
        params = []

        # 遍历 self.train_params，按照其顺序返回相应的属性
        if self.train_params is not None:
            for param in self.train_params:
                param_lower = param.lower()  # 将参数转换为小写，确保不区分大小写
                if param_lower == "mass":
                    params.append(self.mass_dimensionsless)
                elif param_lower == "stiffness":
                    params.append(self.stiffness_dimensionsless)
                elif param_lower == "damping":
                    params.append(self.damping_dimensionsless)
                else:
                    raise AttributeError(
                        f"The attribute '{param_lower}' does not exist in the current object. Please check the train_params configuration.")
        # 遍历 self.userdefined_train_params，按照其顺序返回相应的属性
        if self.userdefined_train_params is not None:
            for param in self.userdefined_train_params:
                try:
                    params.append(getattr(self, param))
                except AttributeError:
                    raise AttributeError(
                        f"The attribute '{param}' does not exist in the current object. Please check the train_params configuration.")
        return params
