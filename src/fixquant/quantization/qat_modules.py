import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import warnings

from torch.nn.modules.utils import _pair
from typing import Optional, List, Tuple, Union
from fixquant.quantization.tqt_quantizer import TQTQuantizer


class _QuantizedConvNd(nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed,
                 output_padding, groups, bias, padding_mode):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, transposed, output_padding, groups, bias,
                         padding_mode)

        self.weight_quantizer = TQTQuantizer(bitwidth=8, tensor_type='weight')
        if bias:
            self.bias_quantizer = TQTQuantizer(bitwidth=8, tensor_type='weight')
        self.act_quantizer = TQTQuantizer(bitwidth=8, tensor_type='act')

        self._mod_name = None

    @property
    def is_quantized(self):
        return True

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def extra_repr(self):
        return super().extra_repr() + f", mod_name={self._mod_name}"

    def state_dict(self, *args, prefix='', **kwargs):
        state = super().state_dict(*args, prefix=prefix, **kwargs)
        state[prefix + 'w_quantizer'] = self.weight_quantizer.state_dict()
        state[prefix + 'b_quantizer'] = self.bias_quantizer.state_dict()
        state[prefix + 'a_quantizer'] = self.act_quantizer.state_dict()
        return state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        w_quantizer_key = prefix + 'w_quantizer'
        if w_quantizer_key in state_dict:
            self.weight_quantizer.load_state_dict(state_dict[w_quantizer_key])
            state_dict.pop(w_quantizer_key)

        b_quantizer_key = prefix + 'b_quantizer'
        if b_quantizer_key in state_dict:
            self.bias_quantizer.load_state_dict(state_dict[b_quantizer_key])
            state_dict.pop(b_quantizer_key)

        a_quantizer_key = prefix + 'a_quantizer'
        if a_quantizer_key in state_dict:
            self.act_quantizer.load_state_dict(state_dict[a_quantizer_key])
            state_dict.pop(a_quantizer_key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)


class QuantizedConv2d(_QuantizedConvNd):
    """A Conv2d module attached with FakeQuantizer modules for weight and bias, used for QAT.
    """
    _FLOAT_MODULE = nn.Conv2d

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                         _pair(0), groups, bias, padding_mode, )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            output = F.conv2d( F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                               weight, bias, self.stride, _pair(0), self.dilation, self.groups)

        output = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    def forward(self, input: Tensor) -> Tensor:
        quantized_weight = self.weight_quantizer.forward(self.weight)
        quantized_bias = self.bias_quantizer.forward(self.bias) if self.bias is not None else None

        output = self._conv_forward(input, quantized_weight, quantized_bias)
        output = self.act_quantizer.forward(output)
        return output


    @classmethod
    def from_float(cls, mod):
        """Create a qat module from a float module.
            Args:
              mod: The float module to be quantized.
                   Must be one of type [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        """
        if type(mod) != cls._FLOAT_MODULE:
            warnings.warn('{} is expected to create from {}, but given {}.'.format(
                cls.__name__, cls._FLOAT_MODULE.__name__, type(mod).__name__))

        conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,)
        conv.weight = mod.weight
        conv.bias = mod.bias
        return conv

    def export_quant_info(self):
        # "conv1": {"weight": 8, "bias": 7, "in": 5, "out": 6}, # @TODO Format accordingly
        frac_w = self.weight_quantizer.export_quant_info()[1]
        frac_b = self.bias_quantizer.export_quant_info()[1] if self.bias is not None else 0
        frac_out = self.act_quantizer.export_quant_info()[1]
        return frac_w, frac_b, frac_out


class QuantizedLinear(nn.Linear):
    """A QuantizedLinear module attached with FakeQuantizer module for weight,
        used for quantization aware training.

        The interface is adopted from `torch.nn.Linear`, please see
        https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        for documentation.
    """
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

        self.weight_quantizer = TQTQuantizer(bitwidth=8, tensor_type='weight')
        if bias:
            self.bias_quantizer = TQTQuantizer(bitwidth=8, tensor_type='weight')
        self.act_quantizer = TQTQuantizer(bitwidth=8, tensor_type='act')
        self._mod_name = None

    def forward(self, input):

        qweight = self.weight_quantizer.forward(self.weight)
        qbias = self.bias_quantizer.forward(self.bias) if self.bias is not None else None
        output = F.linear(input, qweight, qbias)
        output = self.act_quantizer.forward(output)
        return output

    @property
    def is_quantized(self):
        return True

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def extra_repr(self):
        return super().extra_repr() + f", mod_name={self._mod_name}"

    @classmethod
    def from_float(cls, mod):
        """Create a quantized module from a float module.
            Args:
              mod: A float module of type torch.nn.Linear.
              quantizer : A quantizer object that has the quantization params for the module.
        """
        assert type(mod) == cls._FLOAT_MODULE, (' qat.' + cls.__name__ +
                                                '.from_float only works for ' +  cls._FLOAT_MODULE.__name__)

        linear = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, )
        linear.weight = mod.weight
        linear.bias = mod.bias
        return linear

    def export_quant_info(self):
        # "conv1": {"weight": 8, "bias": 7, "in": 5, "out": 6}, # @TODO Format accordingly
        frac_w = self.weight_quantizer.export_quant_info()[1]
        frac_b = self.bias_quantizer.export_quant_info()[1] if self.bias is not None else 0
        frac_out = self.act_quantizer.export_quant_info()[1]
        return frac_w, frac_b, frac_out

    def state_dict(self, *args, prefix='', **kwargs):
        state = super().state_dict(*args, prefix=prefix, **kwargs)
        state[prefix + 'w_quantizer'] = self.weight_quantizer.state_dict()
        state[prefix + 'b_quantizer'] = self.bias_quantizer.state_dict()
        state[prefix + 'a_quantizer'] = self.act_quantizer.state_dict()
        return state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        w_quantizer_key = prefix + 'w_quantizer'
        if w_quantizer_key in state_dict:
            self.weight_quantizer.load_state_dict(state_dict[w_quantizer_key])
            state_dict.pop(w_quantizer_key)

        b_quantizer_key = prefix + 'b_quantizer'
        if b_quantizer_key in state_dict:
            self.bias_quantizer.load_state_dict(state_dict[b_quantizer_key])
            state_dict.pop(b_quantizer_key)

        a_quantizer_key = prefix + 'a_quantizer'
        if a_quantizer_key in state_dict:
            self.act_quantizer.load_state_dict(state_dict[a_quantizer_key])
            state_dict.pop(a_quantizer_key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

class QMaxPool2D(torch.nn.MaxPool2d):
    def __init__(self, kernel_size, stride, padding, dilation, return_indices, ceil_mode,):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

        self.quantizer = TQTQuantizer(bitwidth=8, tensor_type='act')
        self._mod_name = None

    def forward(self, input):

        output = super().forward(input)
        output = self.quantizer.forward(output)
        return output

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def extra_repr(self):
        return super().extra_repr() + f", mod_name={self._mod_name},"

    @classmethod
    def from_float(cls, mod):
        maxpool = cls(mod.kernel_size, mod.stride, mod.padding, mod.dilation, mod.return_indices, mod.ceil_mode, )
        return maxpool

    def export_quant_info(self):
        # "conv1": {"weight": 8, "bias": 7, "in": 5, "out": 6}, # @TODO Format accordingly
        frac_out = self.quantizer.export_quant_info()[1]
        return frac_out

    def state_dict(self, *args, prefix='', **kwargs):
        state = super().state_dict(*args, prefix=prefix, **kwargs)
        state[prefix + 'quantizer'] = self.quantizer.state_dict()
        return state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        quantizer_key = prefix + 'quantizer'
        if quantizer_key in state_dict:
            self.quantizer.load_state_dict(state_dict[quantizer_key])
            state_dict.pop(quantizer_key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)


class QAvgPool2d(torch.nn.modules.AvgPool2d):
    def __init__(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, ):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

        self.quantizer = TQTQuantizer(bitwidth=8, tensor_type='act')
        self._mod_name = None

    def forward(self, input):
        output = super().forward(input)
        output = self.quantizer.forward(output)
        return output

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def extra_repr(self):
        return super().extra_repr() + f", mod_name={self._mod_name},"

    @classmethod
    def from_float(cls, mod, qconfig):
        Qavgpool = cls(mod.output_size)
        return Qavgpool

    def export_quant_info(self):
        # "conv1": {"weight": 8, "bias": 7, "in": 5, "out": 6}, # @TODO Format accordingly
        frac_out = self.quantizer.export_quant_info()[1]
        return frac_out

    def state_dict(self, *args, prefix='', **kwargs):
        state = super().state_dict(*args, prefix=prefix, **kwargs)
        state[prefix + 'quantizer'] = self.quantizer.state_dict()
        return state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        quantizer_key = prefix + 'quantizer'
        if quantizer_key in state_dict:
            self.quantizer.load_state_dict(state_dict[quantizer_key])
            state_dict.pop(quantizer_key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)


class QAdaptiveAvgPool2d(torch.nn.modules.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super().__init__(output_size)

        self.quantizer = TQTQuantizer(bitwidth=8, tensor_type='act')
        self._mod_name = None

    def forward(self, input):
        output = super().forward(input)
        output = self.quantizer.forward(output)
        return output

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def extra_repr(self):
        return super().extra_repr() + f", mod_name={self._mod_name}, "


    @classmethod
    def from_float(cls, mod):
        Qavgpool = cls(mod.output_size)
        return Qavgpool

    def export_quant_info(self):
        # "conv1": {"weight": 8, "bias": 7, "in": 5, "out": 6}, # @TODO Format accordingly
        frac_out = self.quantizer.export_quant_info()[1]
        return frac_out

    def state_dict(self, *args, prefix='', **kwargs):
        state = super().state_dict(*args, prefix=prefix, **kwargs)
        state[prefix + 'quantizer'] = self.quantizer.state_dict()
        return state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        quantizer_key = prefix + 'quantizer'
        if quantizer_key in state_dict:
            self.quantizer.load_state_dict(state_dict[quantizer_key])
            state_dict.pop(quantizer_key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)


class QElementwiseAdd(nn.Module):
    def __init__(self):
        super().__init__()

        self.quantizer = TQTQuantizer(bitwidth=8, tensor_type='act')
        self._mod_name = None

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def extra_repr(self):
        return super().extra_repr() + f", mod_name={self._mod_name}, "

    def forward(self, x1, x2):
        output = x1 + x2
        output = self.quantizer.forward(output)
        return output

    def export_quant_info(self):
        # "conv1": {"weight": 8, "bias": 7, "in": 5, "out": 6}, # @TODO Format accordingly
        frac_out = self.quantizer.export_quant_info()[1]
        return frac_out

    def state_dict(self, *args, prefix='', **kwargs):
        state = super().state_dict(*args, prefix=prefix, **kwargs)
        state[prefix + 'quantizer'] = self.quantizer.state_dict()
        return state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        quantizer_key = prefix + 'quantizer'
        if quantizer_key in state_dict:
            self.quantizer.load_state_dict(state_dict[quantizer_key])
            state_dict.pop(quantizer_key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)


class QuantStubC(nn.Module):
    def __init__(self, bitwidth=8, tensor_type='act', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = TQTQuantizer(bitwidth=bitwidth, tensor_type=tensor_type)
        self._mod_name = "QuantStubC"

    @property
    def module_name(self):
        return self._mod_name

    def forward(self, x):
        return self.quantizer.forward(x)

    def export_quant_info(self):
        # "conv1": {"weight": 8, "bias": 7, "in": 5, "out": 6}, # @TODO Format accordingly
        frac_out = self.quantizer.export_quant_info()[1]
        return frac_out

    def state_dict(self, *args, prefix='', **kwargs):
        state = super().state_dict(*args, prefix=prefix, **kwargs)
        state[prefix + 'quantizer'] = self.quantizer.state_dict()
        return state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        quantizer_key = prefix + 'quantizer'
        if quantizer_key in state_dict:
            self.quantizer.load_state_dict(state_dict[quantizer_key])
            state_dict.pop(quantizer_key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

