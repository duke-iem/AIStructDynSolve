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

class PFNN(NN):
    """Parallel fully-connected network that uses independent sub-networks for each
    network output.

    Args:
        layer_sizes: A nested list that defines the architecture of the neural network.
        activation: The activation function to be used in the network.
        kernel_initializer: A callable initializer for the kernel weights.
    """

    def __init__(self,
                 layer_sizes: list[int],
                 activation: str | list[str],
                 kernel_initializer: callable = None,  # Default is None
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

        if len(layer_sizes) <= 1:
            raise ValueError("Must specify both input and output sizes.")
        if not isinstance(layer_sizes[0], int):
            raise ValueError("Input size must be an integer.")
        if not isinstance(layer_sizes[-1], int):
            raise ValueError("Output size must be an integer.")

        n_output = layer_sizes[-1]

        def make_linear(n_input, n_output):
            linear = torch.nn.Linear(n_input, n_output)
            # Apply kernel initializer
            if kernel_initializer is not None:
               kernel_initializer(linear.weight)
               torch.nn.init.zeros_(linear.bias)
            return linear

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes) - 1):
            prev_layer_size = layer_sizes[i - 1]
            curr_layer_size = layer_sizes[i]

            if isinstance(curr_layer_size, (list, tuple)):
                if len(curr_layer_size) != n_output:
                    raise ValueError(
                        "Number of sub-layers should equal the number of network outputs"
                    )
                if isinstance(prev_layer_size, (list, tuple)):
                    # Case for sub-layers in parallel
                    self.layers.append(
                        torch.nn.ModuleList(
                            [make_linear(prev_layer_size[j], curr_layer_size[j]) for j in range(n_output)]
                        )
                    )
                else:  # e.g., 64 -> [8, 8, 8]
                    self.layers.append(
                        torch.nn.ModuleList(
                            [make_linear(prev_layer_size, curr_layer_size[j]) for j in range(n_output)]
                        )
                    )
            else:  # Regular case, no sub-layers
                if not isinstance(prev_layer_size, int):
                    raise ValueError("Cannot rejoin parallel subnetworks after splitting.")
                self.layers.append(make_linear(prev_layer_size, curr_layer_size))

        # Output layer
        if isinstance(layer_sizes[-2], (list, tuple)):
            self.layers.append(
                torch.nn.ModuleList(
                    [make_linear(layer_sizes[-2][j], 1) for j in range(n_output)]
                )
            )
        else:
            self.layers.append(make_linear(layer_sizes[-2], n_output))

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        # Forward pass through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            # Apply activation
            if isinstance(self.activation, list):
                if isinstance(layer, torch.nn.ModuleList):
                    if isinstance(x, list):
                        x = [self.activation[i](f(x_)) for f, x_ in zip(layer, x)]
                    else:
                        x = [self.activation[i](f(x)) for f in layer]
                else:
                    x = self.activation[i](layer(x))
            else:
                if isinstance(layer, torch.nn.ModuleList):
                    if isinstance(x, list):
                        x = [self.activation(f(x_)) for f, x_ in zip(layer, x)]
                    else:
                        x = [self.activation(f(x)) for f in layer]
                else:
                    x = self.activation(layer(x))


        # Output layer
        if isinstance(x, list):
            x = torch.cat([f(x_) for f, x_ in zip(self.layers[-1], x)], dim=1)
        else:
            x = self.layers[-1](x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        return x
