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
import torch.nn.functional as F
from .NN import NN

# Fully-connected neural network.
class FNN(NN):


    def __init__(
        self,
        layer_sizes: list[int],
        activation: str | list[str],
        kernel_initializer: callable = None  # Default is None
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
        # Check if the provided activation(s) are valid
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

        # Define layers
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i]
                )
            )
            # Apply kernel initializer
            if kernel_initializer is not None:
               kernel_initializer(self.linears[-1].weight)
               torch.nn.init.zeros_(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.linears[:-1]):
            x = linear(x)  # Linear transformation
            # Apply activation
            if isinstance(self.activation, list):
                x = self.activation[j](x)
            else:
                x = self.activation(x)
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x