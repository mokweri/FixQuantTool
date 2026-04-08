import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from fixquant.quantization.fix_ops import FixedPointQuantizer, find_fix_pos, to_int_tensor, fake_quantize_tensor


class FxP_QConv2D(nn.Module):
    def __init__(self, weight, bias, stride, padding, dilation, groups, qconfig):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.qconfig = qconfig
        self.frac_in = None
        self.frac_w = None
        self.frac_b = None
        self.frac_out = None
        self._mod_name = None

    def forward(self, x):
        self.frac_in = self.qconfig[self._mod_name]['frac_in'][0]
        self.frac_w = self.qconfig[self._mod_name]['frac_w']
        self.frac_b = self.qconfig[self._mod_name]['frac_b']
        self.frac_out = self.qconfig[self._mod_name]['frac_out']

        output = torch.nn.functional.conv2d(input=fake_quantize_tensor(x,True,8,self.frac_in),
                                            weight=fake_quantize_tensor(self.weight,True,8,self.frac_w),
                                            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        # Add bias
        if self.bias is not None:
            # Add
            fbias = fake_quantize_tensor(self.bias,True,8,self.frac_b)
            output = output + fbias.view(1, -1, 1, 1)

        return output

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    @classmethod
    def from_float(cls, mod, qconfig=None):
        #  weight, bias, stride, padding, dilation, groups, qconfig):
        conv = cls(
            weight=mod.weight,
            bias=mod.bias,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            qconfig=qconfig
        )

        return conv


class FxP_QLinear(nn.Module):
    def __init__(self, weight, bias, qconfig):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.qconfig = qconfig
        self.frac_in = None
        self.frac_w = None
        self.frac_b = None
        self.frac_out = None
        self._mod_name = None

    def forward(self, x):
        self.frac_in = self.qconfig[self._mod_name]['frac_in'][0]
        self.frac_w = self.qconfig[self._mod_name]['frac_w']
        self.frac_b = self.qconfig[self._mod_name]['frac_b']
        self.frac_out = self.qconfig[self._mod_name]['frac_out']

        output = torch.nn.functional.linear(input=fake_quantize_tensor(x,True,8,self.frac_in),
                                            weight=fake_quantize_tensor(self.weight,True,8,self.frac_w))

        # Add bias
        if self.bias is not None:
            bias = fake_quantize_tensor(self.bias,True,8,self.frac_b)
            # Add
            output = output + bias

        return output

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value


    @classmethod
    def from_float(cls, mod, qconfig=None):
        # (self, weight, bias, qconfig):
        conv = cls(
            weight=mod.weight,
            bias=mod.bias,
            qconfig=qconfig
        )
        return conv


class FxP_QMaxPool2D(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,
                 qconfig=None) -> None:
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.qconfig = qconfig

    def forward(self, x):
        return super().forward(fake_quantize_tensor(x,True,8,self.qconfig[self._mod_name]['frac_in'][0]))

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    @classmethod
    def from_float(cls, mod, qconfig=None):
        maxp = cls(
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            return_indices=mod.return_indices,
            ceil_mode=mod.ceil_mode,
            qconfig=qconfig
        )
        return maxp


class FxP_QElementwiseAdd(nn.Module):
    def __init__(self, qconfig):
        super().__init__()
        self.qconfig = qconfig
        self.frac_in1 = None
        self.frac_in2 = None
        self.frac_out = None
        self._mod_name = None

    def forward(self, x1, x2):
        self.frac_in1 = self.qconfig[self._mod_name]['frac_in'][0]
        self.frac_in2 = self.qconfig[self._mod_name]['frac_in'][1]
        self.frac_out = self.qconfig[self._mod_name]['frac_out']

        out = (fake_quantize_tensor(x1,True,8,self.frac_in1) +
               fake_quantize_tensor(x2,True,8,self.frac_in2))

        return out

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value


class FxP_QAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size, qconfig) -> None:
        super().__init__(output_size)
        self.qconfig = qconfig
        self.frac_in = None
        self.frac_out = None
        self._mod_name = None

    def forward(self, x):

        self.frac_in = self.qconfig[self._mod_name]['frac_in'][0]
        self.frac_out = self.qconfig[self._mod_name]['frac_out']

        xx = fake_quantize_tensor(x,True,8,self.frac_in)
        out = super().forward(xx)

        return out

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    @classmethod
    def from_float(cls, mod, qconfig=None):
        avgpool = cls(
            output_size=mod.output_size,
            qconfig=qconfig
        )
        return avgpool


class FxP_QAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None, qconfig=None) -> None:
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        self.qconfig = qconfig
        self.frac_in = None
        self.frac_out = None
        self._mod_name = None

    def forward(self, x):
        self.frac_in = self.qconfig[self._mod_name]['frac_in'][0]
        self.frac_out = self.qconfig[self._mod_name]['frac_out']
        out = super().forward(fake_quantize_tensor(x,True,8,self.frac_in))
        return out

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    @classmethod
    def from_float(cls, mod, qconfig=None):
        avgpool = cls(
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            ceil_mode=mod.ceil_mode,
            count_include_pad=mod.count_include_pad,
            divisor_override=mod.divisor_override,
            qconfig=qconfig
        )
        return avgpool


if __name__ == '__main__':
    # -----    TESTING the FxPConv2D module ----------------------------
    # Initialize parameters for the convolution
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    input_tensor = torch.tensor([[[[11, 3, 9], [12, 2, 3], [10, 5, 4]],
                                  [[4, 6, 7], [5, 0, 3], [6, 5, 2]],
                                  [[7, 8, 4], [5, 3, 1], [5, 12, 8]]]], dtype=torch.float32)
    # input_tensor = torch.randn(1, in_channels, 32, 32)
    frac_in = find_fix_pos(input_tensor, 8, 3, 2)

    weight = torch.tensor(
        [
            # Filter 1
            [[[0.4, 0.1, 0.3], [0.23, 0.5, 0.0], [0.1, 0.4, 0.7]],
             [[0.9, 0.0, 0.2], [0.12, 0.1, 0.3], [0.0, 1.0, 0.3]],  # Channel 2
             [[0.4, 0.5, 0.8], [0.0, 0.1, 0.3], [0.4, 0.3, 1.0]], ],
            # Filter 2
            [[[0.2, 0.4, 0.0], [0.15, 0.6, 1.0], [0.0, 0.2, 0.3]],  # Channel 2
             [[1.0, 0.4, 0.6], [0.3, 0.8, 0.12], [0.2, 0.5, 0.0]],  # Channel 3
             [[0.2, 0.3, 0.1], [0.4, 1.0, 0.0], [0.1, 0.3, 0.2]], ],
            # Filter 3
            [[[0.0, 1.0, 0.2], [0.4, 0.7, 0.3], [0.0, 0.9, 0.4]],  # Channel 2
             [[0.3, 0.4, 0.6], [0.7, 0.3, 0.15], [0.2, 0.4, 0.0]],  # Channel 3
             [[0.0, 1.0, 0.3], [0.4, 0.0, 0.6], [0.3, 0.4, 0.5]], ],
            # Filter 4
            [[[0.1, 0.1, 0.2], [0.3, 0.4, 0.3], [0.0, 0.1, 0.0]],  # Channel 2
             [[1.0, 0.2, 0.4], [0.2, 0.0, 1.0], [0.4, 0.5, 0.0]],  # Channel 3
             [[0.6, 0.7, 0.8], [0.9, 1.0, 0.0], [0.1, 0.0, 0.12]], ],
        ],
        dtype=torch.float32
    )
    # weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    frac_w = find_fix_pos(weight, 8, 3, 2)

    bias = torch.tensor([0, 0, 1, 0], dtype=torch.float32)
    # bias = torch.randn(out_channels)
    frac_b = find_fix_pos(bias, 8, 3, 2)

    weight = fake_quantize_tensor(weight, n_frac=frac_w)
    bias = fake_quantize_tensor(bias, n_frac=frac_b)
    input_tensor = fake_quantize_tensor(input_tensor, n_frac=frac_in)

    # Standard Conv2D layer
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=groups, bias=True)

    with torch.no_grad():
        conv_layer.weight = nn.Parameter(weight)
        conv_layer.bias = nn.Parameter(bias)

    output_standard = conv_layer(input_tensor).detach()
    frac_out = find_fix_pos(output_standard, 8, 3, 2)
    print('Qparams: frac_in={}, frac_w={}, frac_b={}, frac_out={}'.format(frac_in, frac_w, frac_b, frac_out))

    qweight = to_int_tensor(weight, signed=True, n_bits=8, n_frac=frac_w)
    qbias = to_int_tensor(bias, signed=True, n_bits=8, n_frac=frac_b)

    qin = to_int_tensor(input_tensor, signed=True, n_bits=8, n_frac=frac_in)
    qout = to_int_tensor(output_standard, signed=True, n_bits=8, n_frac=frac_out)

    # Custom FxP_QConv2D layer
    qconfig = {
        'conv1': {
            'frac_in': [frac_in],
            'frac_w': frac_w,
            'frac_b': frac_b,
            'frac_out': frac_out
        }
    }
    fxp_qconv_layer = FxP_QConv2D(weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups,
                                  qconfig=qconfig)
    fxp_qconv_layer.module_name = 'conv1'
    # fxp_qconv_layer.quantize_module()

    output_custom = fxp_qconv_layer(qin).detach()

    print(qout)
    print(output_custom)

    # Compare outputs using Mean Squared Error (MSE)
    mse = F.mse_loss(qout, output_custom)
    print('MSE: {}'.format(mse))

    # -----Test FxPLinear ------------------------

    # in_features = 9
    # out_features = 4
    #
    # input_tensor = torch.tensor([[11, 3, 9, 12, 2, 3, 10, 5, 4]], dtype=torch.float32)
    # print(input_tensor.shape)
    # frac_in = find_fix_pos(input_tensor, 8, 3, 2)
    #
    # # Define weight tensor (out_features x in_features)
    # weight = torch.tensor(
    #     [
    #         [0.4, 0.1, 0.3, 0.23, 0.5, 0.0, 0.1, 0.4, 0.7],
    #         [0.9, 0.0, 0.2, 0.12, 0.1, 0.3, 0.0, 1.0, 0.3],
    #         [0.4, 0.5, 0.8, 0.0, 0.1, 0.3, 0.4, 0.3, 1.0],
    #         [0.2, 0.4, 0.0, 0.15, 0.6, 1.0, 0.0, 0.2, 0.3]
    #     ], dtype=torch.float32)
    # print(weight.shape)
    # frac_w = find_fix_pos(weight, 8, 3, 2)
    #
    # bias = torch.tensor([0, 0, 1, 0], dtype=torch.float32)
    # print(bias.shape)
    # frac_b = find_fix_pos(bias, 8, 3, 2)
    #
    # # Apply fake fixed-point quantization
    # weight = fake_quantize_tensor(weight, n_frac=frac_w)
    # bias = fake_quantize_tensor(bias, n_frac=frac_b)
    # input_tensor = fake_quantize_tensor(input_tensor, n_frac=frac_in)
    #
    # # Standard Linear layer
    # linear_layer = nn.Linear(in_features, out_features, bias=True)
    #
    # # Set weight and bias in the linear layer
    # with torch.no_grad():
    #     linear_layer.weight = nn.Parameter(weight)
    #     linear_layer.bias = nn.Parameter(bias)
    #
    # # Get the standard output
    # output_standard = linear_layer(input_tensor).detach().cpu()
    # frac_out = find_fix_pos(output_standard, 8, 3, 2)
    #
    # print('Qparams: frac_in={}, frac_w={}, frac_b={}, frac_out={}'.format(frac_in, frac_w, frac_b, frac_out))
    #
    # # Quantize the tensors to integer fixed-point values
    # qweight = to_int_tensor(weight, signed=True, n_bits=8, n_frac=frac_w)
    # qbias = to_int_tensor(bias, signed=True, n_bits=8, n_frac=frac_b)
    # qin = to_int_tensor(input_tensor, signed=True, n_bits=8, n_frac=frac_in)
    # qout = to_int_tensor(output_standard, signed=True, n_bits=8, n_frac=frac_out)
    #
    # # Custom FxP_QLinear layer
    # qconfig = {
    #     'linear1': {
    #         'in': frac_in,
    #         'weight': frac_w,
    #         'bias': frac_b,
    #         'out': frac_out
    #     }
    # }
    #
    # # Instantiate the custom quantized linear layer
    # fxp_qlinear_layer = FxP_QLinear(qweight, qbias, qconfig=qconfig)
    #
    # # Set the module name to match the qconfig key
    # fxp_qlinear_layer.module_name = 'linear1'
    #
    # # Get the output from the custom quantized linear layer
    # output_custom = fxp_qlinear_layer(qin).detach().cpu()
    #
    # # Compare the outputs
    # print(qout)
    # print(output_custom)
    #
    # # Calculate Mean Squared Error (MSE)
    # mse = F.mse_loss(qout, output_custom)
    # print('MSE: {}'.format(mse))

    # -----Test FxP_QMaxPool2d

    # # Initialize parameters for MaxPool2D
    # kernel_size = 2  # Kernel size for the pooling layer
    # stride = 2  # Stride for the pooling layer
    # padding = 0  # Padding for the pooling layer
    # ceil_mode = False
    #
    # # Define input tensor (1x3x4x4)
    # input_tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
    #                                [5.0, 6.0, 7.0, 8.0],
    #                                [9.0, 10.0, 11.0, 12.0],
    #                                [13.0, 14.0, 15.0, 16.0]],
    #                               [[17.0, 18.0, 19.0, 20.0],
    #                                [21.0, 22.0, 23.0, 24.0],
    #                                [25.0, 26.0, 27.0, 28.0],
    #                                [29.0, 30.0, 31.0, 32.0]],
    #                               [[33.0, 34.0, 35.0, 36.0],
    #                                [37.0, 38.0, 39.0, 40.0],
    #                                [41.0, 42.0, 43.0, 44.0],
    #                                [45.0, 46.0, 47.0, 48.0]]]], dtype=torch.float32)
    #
    # frac_in = find_fix_pos(input_tensor, 8, 3, 2)
    #
    # avgpool_layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)
    # output_standard = avgpool_layer(input_tensor).detach().cpu()
    # frac_out = find_fix_pos(output_standard, 8, 3, 2)
    # output_standard = to_int_tensor(output_standard, n_frac=frac_out)
    #
    # # Quantization configuration
    # qconfig = {
    #     'pool1': {
    #         'in': frac_in,
    #         'out': frac_out,
    #     }
    # }
    #
    # fxp_qavgpool_layer = FxP_QAvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
    #                                     qconfig=qconfig)
    # fxp_qavgpool_layer.module_name = 'pool1'
    #
    # qin = to_int_tensor(input_tensor, signed=True, n_frac=frac_in)
    # qout = fxp_qavgpool_layer(qin).detach().cpu()
    #
    # # Compare outputs using Mean Squared Error (MSE)
    # mse = F.mse_loss(output_standard, qout)
    # print("Standard AdaptiveAvgPool2d Output:\n", output_standard)
    # print("Custom FxP_QAdaptiveAvgPool2d Output:\n", qout)
    # print("MSE between Standard and Custom Outputs: ", mse.item())

    # Test Addd
    # Define two input tensors (for element-wise addition)
    # input_tensor1 = torch.randn((1, 3, 5, 5), dtype=torch.float32)
    # input_tensor2 = torch.randn((1, 3, 5, 5), dtype=torch.float32)
    # input_tensor1 = torch.tensor([[1.0, 2.0, 3.0],
    #                                 [4.0, 5.0, 6.0],
    #                                 [7.0, 6.0, 9.0]], dtype=torch.float32)
    #
    # input_tensor2 = torch.tensor([[1.0, 2.0, 3.0],
    #                               [4.0, 5.0, 6.0],
    #                               [7.0, 6.0, 9.0]], dtype=torch.float32)
    #
    # input_tensor3 = torch.tensor([[0.5, 0.8, 1],
    #                                 [6, 0.1, 1.5],
    #                                 [3.0, 5., 1.0]], dtype=torch.float32)

    # input_tensor1 = torch.randn((1, 3, 255, 255), dtype=torch.float32)
    # input_tensor2 = torch.randn((1, 3, 255, 255), dtype=torch.float32)
    #
    # # Find fractional positions for quantization
    # frac_in1 = find_fix_pos(input_tensor1, 8, 3, 2)
    # frac_in2 = find_fix_pos(input_tensor2, 8, 3, 2)
    #
    # # Standard element-wise addition using PyTorch
    # input_tensor1 = fake_quantize_tensor(input_tensor1, n_frac=frac_in1)
    # input_tensor2 = fake_quantize_tensor(input_tensor2, n_frac=frac_in2)
    #
    # output_standard = input_tensor1 + input_tensor2
    # frac_out = find_fix_pos(output_standard, 8, 3, 2)
    # output_standard = to_int_tensor(output_standard, n_frac=frac_out)
    #
    # print('QConfig frac_in1={}, frac_in2={}, frac_out={}'.format(frac_in1, frac_in2, frac_out))
    #
    # # Define quantization configuration for custom element-wise add layer
    # qconfig = {
    #     'add1': {
    #         'in1': frac_in1,
    #         'in2': frac_in2,
    #         'out': frac_out
    #     }
    # }
    #
    # # Custom FxP_QElementwiseAdd layer
    # fxp_qadd_layer = FxP_QElementwiseAdd(qconfig=qconfig)
    # fxp_qadd_layer.module_name = 'add1'  # Set the module name to match qconfig key
    #
    # # Get the output of the custom quantized element-wise add layer
    # in1 = to_int_tensor(input_tensor1, n_frac=frac_in1)
    # in2 = to_int_tensor(input_tensor2, n_frac=frac_in2)
    #
    # output_custom = fxp_qadd_layer(in1, in2)
    #
    # # Compare the standard and custom outputs using Mean Squared Error (MSE)
    # mse = F.mse_loss(output_standard, output_custom)
    # print("Standard ElementwiseAdd Output:\n", output_standard)
    # print("Custom FxP_QElementwiseAdd Output:\n", output_custom)
    # print("MSE between Standard and Custom Outputs: ", mse.item())
