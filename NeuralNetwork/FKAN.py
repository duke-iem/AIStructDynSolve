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
import numpy as np
from ..config import DEVICE

# Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients.
class FKAN(NN):
    def __init__(
        self,
        layer_sizes: list[int],
        grid_size: int = 10,
        grid_range: list = None,
        addbias=True,
        smooth_initialization=False,
    ):
        super().__init__()
        self.grid_range = grid_range if grid_range is not None else [0.0, 1.0]
        # 构造 FKAN 层的模块列表
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(NaiveFourierKANLayer(layer_sizes[i], layer_sizes[i + 1], self.grid_range[0], self.grid_range[1], grid_size, addbias, smooth_initialization))

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for layer in self.layers:
            x = layer(x)  # 逐层传递
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x

class NaiveFourierKANLayer(NN):
    def __init__(self, inputdim, outdim, grid_min, grid_max, grid_size, addbias=True, smooth_initialization=False):
        super(NaiveFourierKANLayer, self).__init__()
        self.inputdim = inputdim
        self.outdim = outdim
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_size = grid_size + 1
        self.addbias = addbias
        self.Interval = (self.grid_max-self.grid_min)/self.grid_size
        # With smooth_initialization, fourier coefficients are attenuated by the square of their frequency.
        # This makes KAN's scalar functions smooth at initialization.
        # Without smooth_initialization, high grid_maxs will lead to high-frequency scalar functions,
        # with high derivatives and low correlation between similar inputs.
        grid_norm_factor = (torch.arange(self.grid_max) + 1) ** 2 if smooth_initialization else 2 * np.pi * np.sqrt(self.grid_max)

        # The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        # then each coordinates of the output is of unit variance
        # independently of the various sizes
        self.fouriercoeffs = torch.nn.Parameter(torch.randn(2, self.outdim, self.inputdim, self.grid_size) /
                                             (np.sqrt(self.inputdim) * grid_norm_factor))
        if self.addbias:
            self.bias = torch.nn.Parameter(torch.zeros(1, self.outdim))

    # x.shape ( ... , indim )
    # out.shape ( ..., outdim)
    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim)).to(DEVICE)

        k = torch.reshape(torch.arange(self.Interval, self.grid_max + self.Interval, self.Interval), (1, 1, 1, self.grid_size)).to(DEVICE)

        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        # This should be fused to avoid materializing memory
        c = torch.cos(2 * torch.pi * k * xrshp)
        s = torch.sin(2 * torch.pi * k * xrshp)
        # We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them
        y = torch.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        if self.addbias:
            y += self.bias
        # End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = torch.reshape(c,(1,x.shape[0],x.shape[1],self.grid_max))
        s = torch.reshape(s,(1,x.shape[0],x.shape[1],self.grid_max))
        y2 = torch.einsum( "dbik,djik->bj", torch.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = torch.reshape(y, outshape)
        return y