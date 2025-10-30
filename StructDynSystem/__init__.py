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
    "inputEQ",
    "inputEQ2D",
    "inputEQ3D",
    "inputPt",
    "inputPtFunc",
    "StructDynSystemParm",
    "ModalAnalyzer",
]

from .inputEQ import inputEQ
from .inputEQ2D import inputEQ2D
from .inputEQ3D import inputEQ3D
from .inputPt import inputPt
from .inputPtFunc import inputPtFunc
from .StructDynSystemParm import StructDynSystemParm
from .ModalAnalyzer import ModalAnalyzer
