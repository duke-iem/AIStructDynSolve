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
    "Visualizer",
    "Visualizer_Newmark",
    "Visualizer_SVM",
    "Predictor",
]

from .Visualizer import Visualizer
from .Visualizer_Newmark import Visualizer_Newmark
from .Visualizer_SVM import Visualizer_SVM
from .Predictor import Predictor
