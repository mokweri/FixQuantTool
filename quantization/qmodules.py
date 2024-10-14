import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import warnings

from torch.nn.modules.module import T
from torch.nn.modules.utils import _pair
from typing import Optional, List, Tuple, Union
from quantization.fix_ops import FixedPointQuantizer


class _QuantizedConvNd(nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, transposed, output_padding, groups, bias, padding_mode,
                 qconfig):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, transposed, output_padding, groups, bias,
                         padding_mode)
        assert qconfig, 'Runtime qconfig must be provided for quantized module'
        self.quantizer = FixedPointQuantizer(bitwidth=8)
        self.qconfig = qconfig
        self._mod_name = None

        self.weight_quantizer = self.quantizer.get_weight_quantizer('weight')
        if bias:
            self.bias_quantizer = self.quantizer.get_weight_quantizer('bias')

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
        return super().extra_repr() + (f", mod_name={self._mod_name}, "
                                       f"frac_w = {self.qconfig[self._mod_name]['weight']}, "
                                       f"frac_b = {self.qconfig[self._mod_name]['bias']}, "
                                       f"frac_in = {self.qconfig[self._mod_name]['in']}, "
                                       f"frac_out = {self.qconfig[self._mod_name]['out']}")


class _QuantizedConv(_QuantizedConvNd):

    def forward(self, input: Tensor) -> Tensor:
        self.quantizer.calc_frac_in(input)
        if self.training:
            quantized_weight = self.weight_quantizer(self.weight)
            quantized_bias = self.bias_quantizer(self.bias) if self.bias is not None else None
            output = self._conv_forward(input, quantized_weight, quantized_bias)
            output_quantizer = self.quantizer.get_weight_quantizer('out')
            d_out = output_quantizer(output)
            self.update_qconfig()
        else:
            quantized_weight = self.quantizer.quantize(self.weight, self.qconfig[self._mod_name]['weight'])
            quantized_bias = self.quantizer.quantize(self.bias, self.qconfig[self._mod_name]['bias'])
            output = self._conv_forward(input, quantized_weight, quantized_bias)
            d_out = self.quantizer.quantize(output, self.qconfig[self._mod_name]['out'])
        return d_out

    def update_qconfig(self):
        try:
            if self._mod_name not in self.qconfig:
                raise KeyError(f"Module '{self._mod_name}' not found in qconfig.")

            if "weight" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['weight'] = int(self.quantizer.frac_w)
            else:
                raise KeyError(f"Key 'weight' not found in module '{self._mod_name}' configuration.")

            if "bias" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['bias'] = int(self.quantizer.frac_b)
            else:
                raise KeyError(f"Key 'bias' not found in module '{self._mod_name}' configuration.")

            if "in" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['in'] = int(self.quantizer.frac_in)
            else:
                raise KeyError(f"Key 'out' not found in module '{self._mod_name}' configuration.")

            if "out" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['out'] = int(self.quantizer.frac_out)
            else:
                raise KeyError(f"Key 'out' not found in module '{self._mod_name}' configuration.")

        except KeyError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    @classmethod
    def from_float(cls, mod, qconfig):
        """Create a qat module from a float module.
    Args:
      mod: The float module to be quantized.
          Must be one of type [nn.Conv1d, nn.Conv2d, nn.Conv3d]
      qconfig: An object that specifies the quantization config for the model.
    """

        assert qconfig, 'Runtime qconfig must be provided for quantized module'
        if type(mod) != cls._FLOAT_MODULE:
            warnings.warn('{} is expected to create from {}, but given {}.'.format(
                cls.__name__, cls._FLOAT_MODULE.__name__,
                type(mod).__name__))

        conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            qconfig=qconfig)
        conv.weight = mod.weight
        conv.bias = mod.bias
        return conv


class QuantizedConv2d(_QuantizedConv):
    """A Conv2d module attached with FakeQuantizer modules for weight and bias, used for quantization aware training.
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
                 qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, False, _pair(0), groups, bias, padding_mode,
                         qconfig)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            output = F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        output = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class QuantizedLinear(nn.Linear):
    """A QuantizedLinear module attached with FakeQuantizer module for weight,
    used for quantization aware training.

    The interface is adopted from `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.
    """
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias=True, qconfig=None):
        super().__init__(in_features, out_features, bias)
        assert qconfig, 'qconfig must be provided for quantized module'
        self.quantizer = FixedPointQuantizer(bitwidth=8)
        self.qconfig = qconfig
        self._mod_name = None

        self.weight_quantizer = self.quantizer.get_weight_quantizer('weight')
        if bias:
            self.bias_quantizer = self.quantizer.get_weight_quantizer('bias')

    def forward(self, input):
        self.quantizer.calc_frac_in(input)
        if self.training:
            qweight = self.weight_quantizer(self.weight)
            qbias = self.bias_quantizer(self.bias) if self.bias is not None else None
            output = F.linear(input, qweight, qbias)
            out_quantizer = self.quantizer.get_weight_quantizer('out')
            q_out = out_quantizer(output)
            self.update_qconfig()
        else:
            qweight = self.quantizer.quantize(self.weight, self.qconfig[self._mod_name]['weight'])
            qbias = self.quantizer.quantize(self.bias, self.qconfig[self._mod_name]['bias'])
            output = F.linear(input, qweight, qbias)
            q_out = self.quantizer.quantize(output, self.qconfig[self._mod_name]['out'])
        return q_out

    def update_qconfig(self):
        try:
            if self._mod_name not in self.qconfig:
                raise KeyError(f"Module '{self._mod_name}' not found in qconfig.")

            if "weight" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['weight'] = int(self.quantizer.frac_w)
            else:
                raise KeyError(f"Key 'weight' not found in module '{self._mod_name}' configuration.")

            if "bias" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['bias'] = int(self.quantizer.frac_b)
            else:
                raise KeyError(f"Key 'bias' not found in module '{self._mod_name}' configuration.")

            if "in" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['in'] = int(self.quantizer.frac_in)
            else:
                raise KeyError(f"Key 'in' not found in module '{self._mod_name}' configuration.")

            if "out" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['out'] = int(self.quantizer.frac_out)
            else:
                raise KeyError(f"Key 'out' not found in module '{self._mod_name}' configuration.")

        except KeyError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

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
        return super().extra_repr() + (f", mod_name={self._mod_name}, "
                                       f"frac_w = {self.qconfig[self._mod_name]['weight']}, "
                                       f"frac_b = {self.qconfig[self._mod_name]['bias']}, "
                                       f"frac_in = {self.qconfig[self._mod_name]['in']}, "
                                       f"frac_out = {self.qconfig[self._mod_name]['out']}")

    @classmethod
    def from_float(cls, mod, qconfig):
        """Create a quantized module from a float module.
    Args:
      mod: A float module of type torch.nn.Linear.
      quantizer : A quantizer object that has the quantization params for the module.
    """

        assert qconfig, 'qconfig must be provided for quantized module'
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
                                               cls._FLOAT_MODULE.__name__

        linear = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            qconfig=qconfig)
        linear.weight = mod.weight
        linear.bias = mod.bias
        return linear


class QMaxPool2D(torch.nn.MaxPool2d):
    def __init__(self, kernel_size, stride, padding, dilation, return_indices, ceil_mode, qconfig=None):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

        assert qconfig, 'Runtime qconfig must be provided for quantized module'
        self.quantizer = FixedPointQuantizer(bitwidth=8)
        self.qconfig = qconfig
        self._mod_name = None

    def forward(self, input):
        self.quantizer.calc_frac_in(input)
        output = super().forward(input)
        if self.training:
            out_quantizer = self.quantizer.get_weight_quantizer('out')
            d_out = out_quantizer(output)
            self.update_qconfig()
        else:
            d_out = self.quantizer.quantize(input, self.qconfig[self._mod_name]['out'])
        return d_out

    def update_qconfig(self):
        try:
            if self._mod_name not in self.qconfig:
                raise KeyError(f"Module '{self._mod_name}' not found in qconfig.")

            if "in" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['in'] = int(self.quantizer.frac_in)
            else:
                raise KeyError(f"Key 'in' not found in module '{self._mod_name}' configuration.")

            if "out" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['out'] = int(self.quantizer.frac_out)
            else:
                raise KeyError(f"Key 'out' not found in module '{self._mod_name}' configuration.")

        except KeyError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def extra_repr(self):
        return super().extra_repr() + (f", mod_name={self._mod_name}, "
                                       f"frac_in = {self.qconfig[self._mod_name]['in']}, "
                                       f"frac_out = {self.qconfig[self._mod_name]['out']}")

    @classmethod
    def from_float(cls, mod, qconfig):
        maxpool = cls(mod.kernel_size, mod.stride, mod.padding, mod.dilation, mod.return_indices, mod.ceil_mode,
                      qconfig=qconfig
                      )
        return maxpool


class QAvgPool2d(torch.nn.modules.AvgPool2d):
    def __init__(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override,
                 qconfig=None):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        assert qconfig, 'Runtime qconfig must be provided for quantized module'
        self.quantizer = FixedPointQuantizer(bitwidth=8)
        self.qconfig = qconfig
        self._mod_name = None

    def forward(self, input):
        self.quantizer.calc_frac_in(input)
        output = super().forward(input)
        if self.training:
            out_quantizer = self.quantizer.get_weight_quantizer('out')
            d_out = out_quantizer(output)
            self.update_qconfig()
        else:
            d_out = self.quantizer.quantize(output, self.qconfig[self._mod_name]['out'])
        return d_out

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def extra_repr(self):
        return super().extra_repr() + (f", mod_name={self._mod_name}, "
                                       f"frac_in = {self.qconfig[self._mod_name]['in']}, "
                                       f"frac_out = {self.qconfig[self._mod_name]['out']}")

    def update_qconfig(self):
        try:
            if self._mod_name not in self.qconfig:
                raise KeyError(f"Module '{self._mod_name}' not found in qconfig.")

            if "in" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['in'] = int(self.quantizer.frac_in)
            else:
                raise KeyError(f"Key 'in' not found in module '{self._mod_name}' configuration.")

            if "out" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['out'] = int(self.quantizer.frac_out)
            else:
                raise KeyError(f"Key 'out' not found in module '{self._mod_name}' configuration.")

        except KeyError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    @classmethod
    def from_float(cls, mod, qconfig):
        Qavgpool = cls(mod.output_size, qconfig=qconfig)
        return Qavgpool


class QAdaptiveAvgPool2d(torch.nn.modules.AdaptiveAvgPool2d):
    def __init__(self, output_size, qconfig=None):
        super().__init__(output_size)
        assert qconfig, 'Runtime quantizer must be provided for quantized module'
        self.quantizer = FixedPointQuantizer(bitwidth=8)
        self.qconfig = qconfig
        self._mod_name = None

    def forward(self, input):
        self.quantizer.calc_frac_in(input)
        output = super().forward(input)
        if self.training:
            out_quantizer = self.quantizer.get_weight_quantizer('out')
            d_out = out_quantizer(output)
            self.update_qconfig()
        else:
            d_out = self.quantizer.quantize(output, self.qconfig[self._mod_name]['out'])
        return d_out

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def extra_repr(self):
        return super().extra_repr() + (f", mod_name={self._mod_name}, "
                                       f"frac_in = {self.qconfig[self._mod_name]['in']}, "
                                       f"frac_out = {self.qconfig[self._mod_name]['out']}")

    def update_qconfig(self):
        try:
            if self._mod_name not in self.qconfig:
                raise KeyError(f"Module '{self._mod_name}' not found in qconfig.")

            if "in" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['in'] = int(self.quantizer.frac_in)
            else:
                raise KeyError(f"Key 'in' not found in module '{self._mod_name}' configuration.")

            if "out" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['out'] = int(self.quantizer.frac_out)
            else:
                raise KeyError(f"Key 'out' not found in module '{self._mod_name}' configuration.")

        except KeyError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

    @classmethod
    def from_float(cls, mod, qconfig):
        Qavgpool = cls(mod.output_size, qconfig=qconfig)
        return Qavgpool


class QElementwiseAdd(nn.Module):
    def __init__(self, qconfig=None):
        super().__init__()
        assert qconfig, 'Runtime quantizer must be provided for quantized module'
        self.quantizer = FixedPointQuantizer(bitwidth=8)
        self.qconfig = qconfig
        self._mod_name = None

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def extra_repr(self):
        return super().extra_repr() + (f", mod_name={self._mod_name}, "
                                       f"frac_in1 = {self.qconfig[self._mod_name]['in1']}, "
                                       f"frac_in2 = {self.qconfig[self._mod_name]['in2']}, "
                                       f"frac_out = {self.qconfig[self._mod_name]['out']}")

    def forward(self, x1, x2):
        frac1 = int(self.quantizer.determine_frac(x1))
        frac2 = int(self.quantizer.determine_frac(x2))
        output = x1 + x2
        if self.training:
            out_quantizer = self.quantizer.get_weight_quantizer('out')
            d_out = out_quantizer(output)
            self.update_qconfig(frac1, frac2)
        else:
            d_out = self.quantizer.quantize(output, self.qconfig[self._mod_name]['out'])
        return d_out

    def update_qconfig(self, in1, in2):
        try:
            if self._mod_name not in self.qconfig:
                raise KeyError(f"Module '{self._mod_name}' not found in qconfig.")

            if "in1" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['in1'] = int(in1)
            else:
                raise KeyError(f"Key 'in1' not found in module '{self._mod_name}' configuration.")

            if "in2" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['in2'] = int(in2)
            else:
                raise KeyError(f"Key 'in1' not found in module '{self._mod_name}' configuration.")

            if "out" in self.qconfig[self._mod_name]:
                self.qconfig[self._mod_name]['out'] = int(self.quantizer.frac_out)
            else:
                raise KeyError(f"Key 'out' not found in module '{self._mod_name}' configuration.")

        except KeyError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
