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
import os

# 定义一个映射，将字符串转换为 PyTorch 数据类型
dtype_mapping = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}
# 设置默认数据精度类型，从环境变量读取，若未设置则使用默认值
dtype_str = os.getenv("set_torch_dtype", "torch.float32")  # 默认值为 "torch.float32"
DEFAULT_DTYPE = dtype_mapping.get(dtype_str, torch.float32)  # 映射为 PyTorch 数据类型
print("Current torch dtype is:",DEFAULT_DTYPE)

# 设置默认设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Current device is:",DEVICE)


