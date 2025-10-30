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
from .NN import NN
import torch.nn.functional as F
from ..config import DEVICE

class MsPFFN(NN):
    """
    Multi-scale Parallel Fourier Feature Network (MsPFFN)

    每个输出维度对应一个独立的子网络，支持不同的激活函数和 Fourier 特征频率。

    Args:
        layer_sizes: list, e.g. [1, [64]*3, [64]*3, ..., [64]*3, 3]
            - 每层是 list，表示每个输出维度对应的隐藏层节点数。
        activation: str or list[str], 每个输出维度可设定不同的激活函数
        sigmas: list[float], Fourier 特征的标准差
        kernel_initializer: callable, 可选的权重初始化器
    """
    def __init__(
        self,
        layer_sizes: list,
        activation: str | list[str],
        sigmas: list[float],
        kernel_initializer: callable = None,
    ):
        super().__init__()

        self.sigmas = sigmas
        self.input_dim = layer_sizes[0]
        self.output_dim = layer_sizes[-1]
        self.num_hidden_layers = len(layer_sizes) - 2  # 不含 input 和 output 层

        # 校验隐藏层格式是否为多维列表
        hidden_layers = layer_sizes[1:-1]
        if not all(isinstance(layer, list) for layer in hidden_layers):
            raise ValueError("Each hidden layer spec should be a list[int], one per output dimension.")

        if len(hidden_layers) != self.num_hidden_layers:
            raise ValueError("Mismatch in number of hidden layers.")

        # 激活函数映射
        self.activations_dict = {
            'sin': torch.sin,
            'cos': torch.cos,
            'tan': torch.tan,
            'asin': torch.asin,
            'acos': torch.acos,
            'atan': torch.atan,
            'sinh': torch.sinh,
            'cosh': torch.cosh,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'relu': F.relu,
            'softplus': F.softplus
        }

        # 激活函数处理
        if isinstance(activation, str):
            if activation not in self.activations_dict:
                raise ValueError(f"Invalid activation function: {activation}")
            self.activations = [self.activations_dict[activation]] * self.output_dim
        elif isinstance(activation, list):
            if len(activation) != self.output_dim:
                raise ValueError("Length of activation list must match output_dim.")
            self.activations = [self.activations_dict[act] for act in activation]
        else:
            raise ValueError("Invalid activation format")

        # 构建子网络，每个子网接收 layer_sizes 为纯整数列表
        self.subnets = torch.nn.ModuleList()
        for out_idx in range(self.output_dim):
            # 每个子网的层数为 [input_dim, 每一层的第 out_idx 个数, ..., 1]
            subnet_layer_sizes = [self.input_dim] + [
                layer[out_idx] for layer in hidden_layers
            ] + [1]
            subnet = SingleMsPFFN(
                subnet_layer_sizes,
                self.activations[out_idx],
                sigmas,
                kernel_initializer,
                out_idx
            )
            self.subnets.append(subnet)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        outputs = []
        for subnet in self.subnets:
            y = subnet(x)
            outputs.append(y)

        y_cat = torch.cat(outputs, dim=1)

        if self._output_transform is not None:
            y_cat = self._output_transform(inputs, y_cat)

        return y_cat

class SingleMsPFFN(torch.nn.Module):
    """
    子网络模块：对应 MsPFFN 的每一个输出维度。
    使用 Fourier 特征和多尺度融合。
    """
    def __init__(self, layer_sizes, activation, sigmas, kernel_initializer, out_idx):
        super().__init__()
        self.out_idx = out_idx
        self.activation = activation
        self.sigmas = sigmas

        self.input_dim = layer_sizes[0]

        # Fourier projection weights（多尺度）
        self.b = []
        for sigma in self.sigmas:
            b = torch.nn.Parameter(
                torch.normal(mean=0, std=sigma, size=(layer_sizes[0], layer_sizes[1] // 2), device=DEVICE),
                requires_grad=False  # freeze parameters
            )
            self.b.append(b)

        # 隐藏层
        self.linears = torch.nn.ModuleList()
        for i in range(2, len(layer_sizes) - 1):
            linear = torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            # Apply  kernel initializer
            if kernel_initializer is not None:
                kernel_initializer(linear.weight) # kernel_initializer equivalent
                torch.nn.init.zeros_(linear.bias)
            self.linears.append(linear)

        # 最后一层将所有 Fourier scale 输出融合
        self.dense = torch.nn.Linear(layer_sizes[-2] * len(sigmas), 1)
        if kernel_initializer:
            kernel_initializer(self.dense.weight)
            torch.nn.init.zeros_(self.dense.bias)

    def forward(self, x):
        # Fourier feature layer
        yb = [
            self._fourier_feature_forward(x, self.b[i])
            for i in range(len(self.sigmas))
        ]
        y = [elem[0] for elem in yb]
        self.fourier_feature_weights = [elem[1] for elem in yb]

        # Fully connected layers
        y = [self._fully_connected_forward(_y) for _y in y]

        # Concatenate all the Fourier features
        y = torch.cat(y, dim=1)
        y = self.dense(y)

        return y
    @staticmethod
    def _fourier_feature_forward(y, b):
        y = torch.cat(
            [
                torch.cos(torch.matmul(y, b)),
                torch.sin(torch.matmul(y, b))
            ],
            dim=1
        )
        return y, b

    def _fully_connected_forward(self, y):
        for linear in self.linears:
            y = self.activation(linear(y))
        return y