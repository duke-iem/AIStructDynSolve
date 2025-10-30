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
    "NN",
    "FNN",
    "PFNN",
    "MaFNN",
    "MaPFNN",
    "MsFFN",
    "MsPFFN",
    "KAN",
    "FKAN",
    "RBFKAN",
    "HardBC",
]

from .NN import NN
from .FNN import FNN
from .PFNN import PFNN
from .MaFNN import MaFNN
from .MaPFNN import MaPFNN
from .MsFFN import MsFFN
from .MsPFFN import MsPFFN
from .KAN import KAN
from .FKAN import FKAN
from .RBFKAN import RBFKAN
from .HardBC import HardBC