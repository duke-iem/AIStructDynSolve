"""
AIStructDynSolve
A framework focused on solving structural dynamics problems using artificial intelligence (AI) methods
solve the following ODE of MDOF
M*U_dotdot+C*U_dot+K*U=Pt
initial condition: U(t=0)=InitialU, and U_dot(t=0)=InitialU_dot
Author: 杜轲 duke@iem.ac.cn
Date: 2023/12/26
"""

__all__ = [
    "LossFunctions",
    "NeuralNetwork",
    "StructDynSystem",
    "Trainer",
    "Postprocessor",
    "SVM",
    "Newmark",
    "DEVICE",
    "DEFAULT_DTYPE",
]

from . import LossFunctions
from . import NeuralNetwork
from . import StructDynSystem
from . import Trainer
from . import Postprocessor
from . import SVM
from . import Newmark
from .config import DEFAULT_DTYPE
from .config import DEVICE
import torch

# 初始化默认数据类型
torch.set_default_dtype(DEFAULT_DTYPE)
torch.set_default_device(DEVICE)


