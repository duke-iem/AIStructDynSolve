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

"""Multi-scale Fourier feature networks.
    Args:
        sigmas: List of standard deviations of the distribution of Fourier feature
            embeddings.
    """
class MsFFN(NN):

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str | list[str],
        sigmas: list[float],
        kernel_initializer: callable = None,  # Default is None,
    ):
        super().__init__()

        # Dictionary of allowed activations
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

        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            # Use activations_dict to get each activation function
            self.activation = [self.activations_dict[act] for act in activation if act in self.activations_dict]
        else:
            # Single activation function for all layers
            if activation not in self.activations_dict:
                raise ValueError(f"Invalid activation function: {activation}")
            self.activation = self.activations_dict[activation]  # Single activation function

        self.sigmas = sigmas  # list or tuple
        self.fourier_feature_weights = None

        self.b = []
        for sigma in self.sigmas:
            b = torch.nn.Parameter(
                torch.normal(mean=0, std=sigma, size=(layer_sizes[0], layer_sizes[1] // 2), device=DEVICE),
                requires_grad=False  # freeze parameters
            )
            self.b.append(b)

        self.linears = torch.nn.ModuleList()
        for i in range(2, len(layer_sizes) - 1):
            linear = torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            # Apply  kernel initializer
            if kernel_initializer is not None:
                kernel_initializer(linear.weight) # kernel_initializer equivalent
                torch.nn.init.zeros_(linear.bias)
            self.linears.append(linear)

        self._dense = torch.nn.Linear(layer_sizes[-2] * len(self.sigmas), layer_sizes[-1])
        # Apply  kernel initializer
        if kernel_initializer is not None:
            kernel_initializer(self._dense.weight)
            torch.nn.init.zeros_(self._dense.bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

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
        y = self._dense(y)

        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
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
