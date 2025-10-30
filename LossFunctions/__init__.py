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
    "Gradients",
    "LossEQ",
    "LossEQ2D",
    "LossEQ3D",
    "LossPt",
    "LossPtFunc",
]

from .Gradients import Gradients
from .LossEQ import LossEQ
from .LossEQ2D import LossEQ2D
from .LossEQ3D import LossEQ3D
from .LossPt import LossPt
from .LossPtFunc import LossPtFunc





