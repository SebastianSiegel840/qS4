import numpy as np
import torch

from torch.nn import Linear
from torch.nn import functional as F
from torch import Tensor

# Precision reduction for numpy matrices #
def reduce_precision_np(M, prec=2**16):
    max = np.max(M)
    min = np.min(M)

    vals = np.linspace(min, max, prec)
    #print(M.shape)
    if len(M.shape) == 2:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M[i, j] = vals[np.argmin(np.power(vals - M[i, j], 2))]
    elif len(M.shape) == 1:
        for i in range(M.shape[0]):
            M[i] = vals[np.argmin(np.power(vals - M[i], 2))]
    return M

# Precision reduction for pytorch tensors #
def reduce_precision_t(M, prec=2**16):
    max = torch.max(M)
    min = torch.min(M)
    device = M.device

    vals = torch.linspace(min, max, prec).to(device)
    #print(M.shape)
    if len(M.shape) == 2:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M[i, j] = vals[torch.argmin(torch.pow(vals - M[i, j], 2))]
    elif len(M.shape) == 1:
        for i in range(M.shape[0]):
            M[i] = vals[torch.argmin(torch.pow(vals - M[i], 2))]
    return M


class MaxQuantFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, quant_levels=2, defmax=None):
        #print("Forward called!")
        #a = a.clone()
        #quant_levels = quant_levels.clone()
        if defmax is not None:
            ctx.save_for_backward(a, torch.tensor(quant_levels), torch.tensor(defmax))
        else:
            ctx.save_for_backward(a, torch.tensor(quant_levels), torch.tensor(0.0))


        if defmax is None:
            scale = quant_levels / 2 / torch.clamp(torch.max(a.abs().flatten(), dim=-1, keepdim=True)[0], min=1e-5) 
        else:
            scale = quant_levels / 2 / torch.clamp(defmax, min=1e-5)
        a_out = torch.clamp((a * scale).round(), min=-quant_levels // 2, max=quant_levels // 2) / scale
        return a_out

    @staticmethod
    def backward(ctx, grad_out):
        #qa_approx = a - torch.sin(a / delta * torch.pi)**3

        a, quant_levels, defmax = ctx.saved_tensors

        k = 3
        delta = torch.max(torch.abs(a)) / quant_levels

        sur_grad = k * torch.sin(a / delta * torch.pi) ** (k-1) * torch.cos(a / delta * torch.pi) / delta * torch.pi

        grad_input = grad_out.clone() * sur_grad
        return grad_input, None, None


# Quantization for quantization-wawre training with STE
# taken from bitnet 1.58b
def max_quant_fn(a, quant_levels=2):
        # scaling parameter to get an estimate of the magnitude of the activations. 
    # clamp to avoid division by zero
    #import pdb
    #pdb.set_trace()
    scale = quant_levels / 2 / torch.clamp(torch.max(a.abs().flatten(), dim=-1, keepdim=True)[0], min=1e-5) 

    # a * scale normalizes a. rounding brings them to the next integer. 
    # clamping to cut off values above the quantization limits. / scale to undo normalization
    a_out = torch.clamp((a * scale).round(), min=-quant_levels // 2 + (quant_levels + 1)%2, max=quant_levels // 2) / scale
    return a_out

# taken from bitnet 1.58b
def mean_quant_fn(w, quant_levels=2):
    # scaling parameter to get an estimate of the magnitude of the weights. 
    # clamp to avoid division by zero
    scale = quant_levels / 2 / w.abs().flatten().mean().clamp(min=1e-5) 

    # w * scale normalizes w. rounding brings them to the next integer. 
    # clamping to cut off values above the quantization limits. / scale to undo normalization
    w_out = (w * scale).round().clamp(-quant_levels // 2, quant_levels // 2) / scale
    #w_out = (w * scale).round().clamp(-quant_levels // 2 + (quant_levels + 1)%2, quant_levels // 2) / scale
    return w_out


class QuantizedLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, quant_levels=None, quant_fn=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.quant_levels = quant_levels
        self.quant_fn = quant_fn
    
    def forward(self, input: Tensor) -> Tensor:
        # return F.linear(input, self.weight - (self.weight - self.quant_fn(self.weight, self.quant_levels)).detach(), self.bias)
        return F.linear(input, MaxQuantFn.apply(self.weight, self.quant_levels), self.bias)

    def analysis(self):
        return self.weight, MaxQuantFn.apply(self.weight, self.quant_levels)
    

class BaseLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
    
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
    
    def analysis(self):
        return self.weight, self.weight # double export to match analysis of quantized layer
    

from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.nn import Module
import math
    
class Linear(Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    

from torch.nn.common_types import _size_1_t
from typing import Union

class QuantizedConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        quant_levels=None, quant_fn=None
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.quant_levels = quant_levels
        self.quant_fn= quant_fn

    def forward(self, input: Tensor) -> Tensor:
        if self.quant_levels is not None:
            return self._conv_forward(input, MaxQuantFn.apply(self.weight, self.quant_levels), self.bias)
        else:
            return self._conv_forward(input, self.weight, self.bias)

    def analysis(self):
        if self.quant_levels is not None:
            return self.weight, self.quant_fn(self.weight, self.quant_levels)
        else:
            return self.weight, self.weight