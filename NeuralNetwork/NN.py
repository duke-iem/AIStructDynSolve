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

"""Base class for all neural network modules."""
class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._input_transform = None
        self._output_transform = None

    def apply_feature_transform(self, transform):
        """Compute the features by applying a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        # TODO: support input transform
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        # TODO: support output transform
        self._output_transform = transform
