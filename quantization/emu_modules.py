import torch
import torch.nn as nn
from typing import Tuple

from quantization.fix_ops import FixedPointQuantizer, round_tensor, find_fix_pos, to_int_tensor, fake_quantize_tensor


class FxP_QConv2D(nn.Module):
    def __init__(self, weight, bias, stride, padding, dilation, groups,
                 qconfig):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.qconfig = qconfig
        self.quantizer = FixedPointQuantizer(bitwidth=8)
        self.frac_in = None
        self.frac_w = None
        self.frac_b = None
        self.frac_out = None
        self._mod_name = None

    def forward(self, x):
        self.frac_in = self.qconfig[self._mod_name]['in']
        self.frac_w = self.qconfig[self._mod_name]['weight']
        self.frac_b = self.qconfig[self._mod_name]['bias']
        self.frac_out = self.qconfig[self._mod_name]['out']

        output = torch.nn.functional.conv2d(input=x.float().cuda(), weight=self.weight.float().cuda(),
                                            stride=self.stride, padding=self.padding, dilation=self.dilation,
                                            groups=self.groups)
        output = output.type(torch.int32)
        output = output / (2 ** (self.frac_w + self.frac_in - self.frac_out))
        output = round_tensor(output, mode='HALF_UP')

        # Add bias
        if self.bias is not None:
            if self.frac_b == self.frac_out:
                bias = self.bias.type(torch.int32)
            elif self.frac_b > self.frac_out:
                bias = self.bias.type(torch.int32) / (2 ** (self.frac_b - self.frac_out))
                bias = round_tensor(bias, mode='HALF_UP')
            else:
                bias = self.bias.type(torch.int32) / (2 ** (self.frac_out - self.frac_b))
                bias = round_tensor(bias, mode='HALF_UP')
            # Add
            output = output + bias.view(1, -1, 1, 1).cuda()

        output = torch.clamp(output, -128, 127)
        return output

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value


class FxP_QLinear(nn.Module):
    def __init__(self, weight, bias, qconfig):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.qconfig = qconfig
        self.quantizer = FixedPointQuantizer(bitwidth=8)
        self.frac_in = None
        self.frac_w = None
        self.frac_b = None
        self.frac_out = None
        self._mod_name = None

    def forward(self, x):
        self.frac_in = self.qconfig[self._mod_name]['in']
        self.frac_w = self.qconfig[self._mod_name]['weight']
        self.frac_b = self.qconfig[self._mod_name]['bias']
        self.frac_out = self.qconfig[self._mod_name]['out']

        output = torch.nn.functional.linear(input=x.float().cuda(), weight=self.weight.float().cuda())
        output = output.type(torch.int32)
        output = output / (2 ** (self.frac_w + self.frac_in - self.frac_out))
        output = round_tensor(output, mode='HALF_UP')

        # Add bias
        if self.bias is not None:
            if self.frac_b == self.frac_out:
                bias = self.bias.type(torch.int32)
            elif self.frac_b > self.frac_out:
                bias = self.bias.type(torch.int32) / (2 ** (self.frac_b - self.frac_out))
                bias = round_tensor(bias, mode='HALF_UP')
            else:
                bias = self.bias.type(torch.int32) / (2 ** (self.frac_out - self.frac_b))
                bias = round_tensor(bias, mode='HALF_UP')
            # Add
            output = output + bias.view(1, -1, 1, 1).cuda()

        output = torch.clamp(output, -128, 127)
        return output

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value


if __name__ == '__main__':
    # -----TESTING
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
                                  [[7, 8, 4], [5, 3, 1], [5, 12, 8]]]], dtype=torch.float32).cuda()
    # input_tensor = torch.randn(1, in_channels, 32, 32).cuda()
    frac_in = find_fix_pos(input_tensor, 8, 3, 2).int()

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
    ).cuda()
    # weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()

    frac_w = find_fix_pos(weight, 8, 3, 2).int()

    bias = torch.tensor([0, 0, 1, 0], dtype=torch.float32).cuda()
    # bias = torch.randn(out_channels).cuda()
    frac_b = find_fix_pos(bias, 8, 3, 2).int()

    weight = fake_quantize_tensor(weight, n_frac=frac_w)
    bias = fake_quantize_tensor(bias, n_frac=frac_b)
    input_tensor = fake_quantize_tensor(input_tensor, n_frac=frac_in)

    # Standard Conv2D layer
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=groups, bias=True).cuda()


    with torch.no_grad():
        conv_layer.weight = nn.Parameter(weight)
        conv_layer.bias = nn.Parameter(bias)

    output_standard = conv_layer(input_tensor).detach().cpu()
    frac_out = find_fix_pos(output_standard, 8, 3, 2).int()
    print('Qparams: frac_in={}, frac_w={}, frac_b={}, frac_out={}'.format(frac_in, frac_w, frac_b, frac_out))

    qweight = to_int_tensor(weight, signed=True, n_bits=8, n_frac=frac_w)
    qbias = to_int_tensor(bias, signed=True, n_bits=8, n_frac=frac_b)
    qin = to_int_tensor(input_tensor, signed=True, n_bits=8, n_frac=frac_in)
    qout = to_int_tensor(output_standard, signed=True, n_bits=8, n_frac=frac_out)

    # Custom FxP_QConv2D layer
    qconfig = {
        'conv1': {
            'in': frac_in,
            'weight': frac_w,
            'bias': frac_b,
            'out': frac_out
        }
    }
    fxp_qconv_layer = FxP_QConv2D(qweight, qbias, stride=stride, padding=padding, dilation=dilation, groups=groups,
                                  qconfig=qconfig).cuda()

    fxp_qconv_layer.module_name = 'conv1'  # Set the module name to match qconfig key

    output_custom = fxp_qconv_layer(qin).detach().cpu()

    print(qout)
    print(output_custom)

    # Compare outputs using Mean Squared Error (MSE)
    #mse = F.mse_loss(output_custom, output_standard)
