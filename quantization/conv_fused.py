import math
from typing import ClassVar, Optional, Type
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.nn.utils import fuse_conv_bn_weights
from quantization.fix_ops import FixedPointQuantizer
from quantization.qmodules import QuantizedConv2d


# Adopted from https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_optimizer/pytorch_binding/pytorch_nndct/nn
# /quantization/modules/conv_fused.py

_BN_CLASS_MAP = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}

# Number of steps before freezing the batch norm running average and variance
FREEZE_BN_DELAY_DEFAULT = 200000


# Used for identifying intrinsic modules used in quantization
class _FusedModule(torch.nn.Sequential):
    pass


class _ConvBnNd(nn.modules.conv._ConvNd, _FusedModule):
    _version = 2
    _FLOAT_MODULE: ClassVar[Type[nn.modules.conv._ConvNd]]

    def __init__(
            self,
            # ConvNd args
            in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups,
            bias, padding_mode,
            # BatchNormNd args
            eps=1e-05, momentum=0.1,
            # Args for this module
            freeze_bn_delay=FREEZE_BN_DELAY_DEFAULT,
            qconfig=None, dim=2, _mod_name=None):
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels,
                                         kernel_size, stride, padding, dilation,
                                         transposed, output_padding, groups, False,
                                         padding_mode)
        assert qconfig, 'qconfig must be provided for quantized module {}'.format(
            self.__class__.__name__)
        #self.bn_frozen = freeze_bn_stats if self.training else True
        self.bn_frozen = False
        self.freeze_bn_delay = freeze_bn_delay
        self.dim = dim
        self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)
        self.quantizer = FixedPointQuantizer(bitwidth=8)
        self.qconfig = qconfig
        self.weight_quantizer = self.quantizer.get_weight_quantizer('weight')
        self.bias_quantizer = self.quantizer.get_weight_quantizer('bias')
        self._mod_name = None

        # print(qconfig)
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters as they modify the same state
        if self.training:
            if self.bn_frozen:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

        self.conv_bn_fused = False
        self._enable_slow_path_for_better_numerical_stability = True

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super().reset_parameters()

    def update_bn_stats(self):
        self.bn_frozen = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.merge_bn_to_conv()
        self.bn_frozen = True
        self.bn.training = False
        return self

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def _forward(self, input):
        if self._enable_slow_path_for_better_numerical_stability:
            output = self._forward_slow(input)
        output = self._forward_approximate(input)

        if self.training:
            out_quantizer = self.quantizer.get_weight_quantizer('out')
            q_out = out_quantizer(output)
            self.update_qconfig()
        else:
            q_out = self.quantizer.quantize(output, self.qconfig[self._mod_name]['out'])
        return q_out

    def _forward_approximate(self, input):
        """Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        # scaled_weight = self.weight_fake_quant( self.weight * scale_factor.reshape(weight_shape))
        scaled_weight = self.weight_quantizer(self.weight * scale_factor.reshape(weight_shape))

        # using zero bias here since the bias for original conv will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias, dtype=input.dtype)
        else:
            zero_bias = torch.zeros(
                self.out_channels, device=scaled_weight.device, dtype=input.dtype
            )
        conv = self._conv_forward(input, scaled_weight, zero_bias)
        conv_orig = conv / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            conv_orig = conv_orig + self.bias_quantizer(self.bias.reshape(bias_shape))
        conv = self.bn(conv_orig)
        return conv

    def _forward_slow(self, input):
        """
        A more accurate but slow method to compute conv bn fusion, following https://arxiv.org/pdf/1806.08342.pdf
        It requires two forward passes but handles the case bn.weight == 0

        Conv: Y = WX + B_c
        Conv without bias: Y0 = WX = Y - B_c, Y = Y0 + B_c

        Batch statistics:
          mean_Y = Y.mean()
                 = Y0.mean() + B_c
          var_Y = (Y - mean_Y)^2.mean()
                = (Y0 - Y0.mean())^2.mean()
        BN (r: bn.weight, beta: bn.bias):
          Z = r * (Y - mean_Y) / sqrt(var_Y + eps) + beta
            = r * (Y0 - Y0.mean()) / sqrt(var_Y + eps) + beta

        Fused Conv BN training (std_Y = sqrt(var_Y + eps)):
          Z = (r * W / std_Y) * X + r * (B_c - mean_Y) / std_Y + beta
            = (r * W / std_Y) * X - r * Y0.mean() / std_Y + beta

        Fused Conv BN inference (running_std = sqrt(running_var + eps)):
          Z = (r * W / running_std) * X - r * (running_mean - B_c) / running_std + beta

        QAT with fused conv bn:
          Z_train = fake_quant(r * W / running_std) * X * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
                  = conv(X, fake_quant(r * W / running_std)) * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
          Z_inference = conv(X, fake_quant(r * W / running_std)) - r * (running_mean - B_c) / running_std + beta
        """

        assert self.bn.running_var is not None
        assert self.bn.running_mean is not None

        # using zero bias here since the bias for original conv will be added later
        zero_bias = torch.zeros(
            self.out_channels, device=self.weight.device, dtype=input.dtype
        )

        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1

        if self.bn.training:
            # needed to compute batch mean/std
            conv_out = self._conv_forward(input, self.weight, zero_bias)
            # update bn statistics
            with torch.no_grad():
                conv_out_bias = (
                    conv_out
                    if self.bias is None
                    else conv_out + self.bias.reshape(bias_shape)
                )
                self.bn(conv_out_bias)

        # fused conv + bn without bias using bn running statistics
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std

        if self.training:
            scaled_weight = self.weight_quantizer(self.weight * scale_factor.reshape(weight_shape))
        else:
            scaled_weight = self.quantizer.quantize(self.weight * scale_factor.reshape(weight_shape),
                                                    self.qconfig[self._mod_name]['weight'])

        # fused conv without bias for inference: (r * W / running_std) * X
        conv_bn = self._conv_forward(input, scaled_weight, zero_bias)

        if self.bn.training:
            avg_dims = [0] + list(range(2, len(self.weight.shape)))
            batch_mean = conv_out.mean(avg_dims)  # type: ignore[possibly-undefined]
            batch_var = torch.square(conv_out - batch_mean.reshape(bias_shape)).mean(
                avg_dims
            )
            batch_std = torch.sqrt(batch_var + self.bn.eps)

            # scale to use batch std in training mode
            # conv(X, r * W / std_Y) = conv(X, r * W / running_std) * (running_std / std_Y)
            unscale_factor = running_std / batch_std
            conv_bn *= unscale_factor.reshape(bias_shape)

            fused_mean = batch_mean
            fused_std = batch_std
        else:
            fused_mean = self.bn.running_mean - (
                self.bias if self.bias is not None else 0
            )
            fused_std = running_std

        # fused bias = beta - r * mean / std
        fused_bias = self.bn.bias - self.bn.weight * fused_mean / fused_std
        if self.training:
            conv_bn += self.bias_quantizer(fused_bias.reshape(bias_shape))
            self.update_qconfig()
        else:
            conv_bn += self.quantizer.quantize(fused_bias.reshape(bias_shape), self.qconfig[self._mod_name]['bias'])

        # HACK to let conv bias participate in loss to avoid DDP error (parameters
        #   were not used in producing loss)
        if self.bias is not None:
            conv_bn += (self.bias - self.bias).reshape(bias_shape)

        return conv_bn

    def extra_repr(self):
        return super().extra_repr() + (f", mod_name={self._mod_name}, "
                                       f"frac_w = {self.qconfig[self._mod_name]['weight']}, "
                                       f"frac_b = {self.qconfig[self._mod_name]['bias']}, "
                                       f"frac_in = {self.qconfig[self._mod_name]['in']}, "
                                       f"frac_out = {self.qconfig[self._mod_name]['out']}")

    # check out https://github.com/IntelLabs/distiller/blob/master/distiller/quantization/sim_bn_fold.py#L87
    def forward(self, input):
        self.quantizer.calc_frac_in(input)
        return self._forward(input)

    def merge_bn_to_conv(self):
        self.conv_bn_fused = True

    def train(self, mode=True):
        """Batchnorm's training behavior is using the self.training flag.
        Prevent changing it if BN is frozen. This makes sure that calling `model.train()` on a model with a frozen BN
        will behave properly.
    """
        self.training = mode
        if not self.bn_frozen:
            for module in self.children():
                module.train(mode)
        return self

    @property
    def is_quantized(self):
        return True

    @property
    def node_name(self):
        return self.node_name

    @node_name.setter
    def node_name(self, name):
        self.node_name = name

    def update_qconfig(self):
        try:
            if self._mod_name not in self.qconfig:
                raise KeyError(f"Module '{self.mod_name}' not found in qconfig.")

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

    def quantize_module(self):
        self.weight = nn.Parameter(self.quantizer.quantize(self.weight, self.qconfig[self._mod_name]['weight']))
        if self.bias is not None:
            self.bias = nn.Parameter(self.quantizer.quantize(self.bias, self.qconfig[self._mod_name]['bias']))

    # ===== Serialization version history =====
    #
    # Version 1/None
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- gamma : Tensor
    #   |--- beta : Tensor
    #   |--- running_mean : Tensor
    #   |--- running_var : Tensor
    #   |--- num_batches_tracked : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- bn : Module
    #        |--- weight : Tensor (moved from v1.self.gamma)
    #        |--- bias : Tensor (moved from v1.self.beta)
    #        |--- running_mean : Tensor (moved from v1.self.running_mean)
    #        |--- running_var : Tensor (moved from v1.self.running_var)
    #        |--- num_batches_tracked : Tensor (moved from v1.self.num_batches_tracked)
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs,):
        version = local_metadata.get("version", None)
        if version is None or version == 1:
            # BN related parameters and buffers were moved into the BN module for v2
            v2_to_v1_names = {
                "bn.weight": "gamma",
                "bn.bias": "beta",
                "bn.running_mean": "running_mean",
                "bn.running_var": "running_var",
                "bn.num_batches_tracked": "num_batches_tracked",
            }
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif prefix + v2_name in state_dict:
                    # there was a brief period where forward compatibility for this module was broken (between
                    # https://github.com/pytorch/pytorch/pull/38478
                    # and https://github.com/pytorch/pytorch/pull/38820)
                    # and modules emitted the v2 state_dict format while specifying that version == 1.
                    # This patches the forward compatibility issue by allowing the v2 style entries to be used.
                    pass
                elif strict:
                    missing_keys.append(prefix + v2_name)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs,)

    @classmethod
    def from_float(cls, conv, bn, qconfig):
        """Create a qat module from a float module."""
        assert qconfig, 'Input float module must have a valid qconfig'

        convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                     conv.groups,
                     conv.bias is not None, conv.padding_mode, bn.eps, bn.momentum,
                     False, qconfig)
        convbn.weight = conv.weight
        convbn.bias = conv.bias
        convbn.bn.weight = bn.weight
        convbn.bn.bias = bn.bias
        convbn.bn.running_mean = bn.running_mean
        convbn.bn.running_var = bn.running_var
        convbn.bn.num_batches_tracked = bn.num_batches_tracked
        convbn.bn.eps = bn.eps
        return convbn

    def to_float(self):
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
        )
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())

        if cls._FLOAT_BN_MODULE:
            # fuse bn into conv
            assert self.bn.running_var is not None and self.bn.running_mean is not None
            conv.weight, conv.bias = fuse_conv_bn_weights(
                conv.weight,
                conv.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias,
            )
        conv.train(self.training)
        return conv

    def to_fusedQConv2d(self):
        conv = QuantizedConv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
            self.qconfig
        )
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())

        # fuse bn into conv
        assert self.bn.running_var is not None and self.bn.running_mean is not None
        conv.weight, conv.bias = fuse_conv_bn_weights(
            conv.weight,
            conv.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps,
            self.bn.weight,
            self.bn.bias,
        )
        conv.module_name = self._mod_name
        return conv


class QuantizedConvBatchNorm2d(_ConvBnNd, nn.Conv2d):
    """A QuantizedConvBatchNorm2d module is a module fused from Conv2d and BatchNorm2d
    attached with FakeQuantizer modules for weight and batchnorm stuffs used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and :class:`torch.nn.BatchNorm2d`.
    Implementation details: https://arxiv.org/pdf/1806.08342.pdf section 3.2.2

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantizer modules initialized to default.
    """

    _FLOAT_MODULE: ClassVar["Type[QuantizedConvBatchNorm2d]"]
    _FLOAT_CONV_MODULE: ClassVar[Type[nn.Conv2d]] = nn.Conv2d
    _FLOAT_BN_MODULE: ClassVar[Optional[Type[nn.Module]]] = nn.BatchNorm2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=None, padding_mode='zeros',
                 # BatchNorm2d args
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn_delay=FREEZE_BN_DELAY_DEFAULT,
                 quantizer=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                           _pair(0), groups, bias, padding_mode, eps, momentum, freeze_bn_delay, quantizer, dim=2)


QuantizedConvBatchNorm2d._FLOAT_MODULE = QuantizedConvBatchNorm2d
_FUSED_CLS = [QuantizedConvBatchNorm2d]


def update_bn_stats(mod):
    if type(mod) in _FUSED_CLS:
        mod.update_bn_stats()


def freeze_bn_stats(mod):
    if type(mod) in _FUSED_CLS:
        mod.freeze_bn_stats()


def fuse_conv_bn(mod):
    if type(mod) in _FUSED_CLS:
        mod.merge_bn_to_conv()


def clear_non_native_bias(mod):
    if type(mod) in _FUSED_CLS:
        mod.clear_non_native_bias()
