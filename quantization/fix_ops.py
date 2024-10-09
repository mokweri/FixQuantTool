import sys
import torch


def fake_quantize_per_tensor(x, scale_inv, zero_point, quant_min, quant_max, method, inplace):
    if method == -1:
        # Affine quantization (automatic handling by PyTorch)
        return torch.fake_quantize_per_tensor_affine(x, 1 / scale_inv, zero_point, quant_min, quant_max)
    else:
        # Custom fixed-point quantization
        if inplace:
            output = x
        else:
            output = x.clone()

        # Scale input to fixed-point representation
        # input_scaled = torch.round(input * scale)
        input_scaled = round_tensor(x * scale_inv, 'HALF_UP')

        # Clamp to ensure values are within the fixed-point range
        output.copy_(torch.clamp(input_scaled, quant_min, quant_max) / scale_inv)

        return output


def find_fix_pos(input, bit_width, scope, method):
    """ An equivalent of the fi() object in matlab"""
    # check if input tensor is all zeros
    if torch.all(torch.isclose(input, torch.zeros_like(input), atol=1e-7)):
        return bit_width - 1

    # get max and min element in the tensor
    abs_max = 1 << (bit_width - 1)
    fix_lb = -abs_max - 0.5
    fix_ub = abs_max - 0.5
    x_max = torch.max(input)
    x_min = torch.min(input)

    # calculate step and fix pos based on max and min value
    step = torch.max(x_min / fix_lb, x_max / fix_ub)
    max_scale = torch.floor(torch.log2(1.0 / step)) if step > sys.float_info.min else torch.tensor(18)

    # calculate step based on diffs
    final_scale = max_scale
    fixed_diff_min = sys.float_info.max
    if scope > 1:
        # avoid clone multiple times
        input = input.clone()
        for i in range(0, scope):
            scale = max_scale + i
            qinput = fake_quantize_per_tensor(input, pow(2.0, scale), 0, -abs_max, abs_max - 1, method, 0)
            qinput = torch.sub(input, qinput)
            qinput = torch.pow(qinput, 2.0)
            diff = torch.sum(qinput).item()
            if diff < fixed_diff_min:
                final_scale = scale
                fixed_diff_min = diff

    return final_scale


def round_tensor(x, mode='HALF_TO_EVEN'):
    """Rounds the tensor x according to the specified mode."""
    if mode == 'HALF_TO_EVEN':
        # Round to the nearest even number
        return torch.where((x - torch.floor(x) == 0.5) & (torch.floor(x) % 2 == 0), torch.floor(x), torch.round(x))
    elif mode == 'HALF_UP':
        # Traditional round half up
        return torch.floor(x + 0.5)
    elif mode == 'HALF_AWAY_FROM_ZERO':
        # Round .5 away from zero
        return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)
    else:
        raise ValueError("Unsupported rounding mode")


class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, min_val, max_val, round_mode, inplace=False):
        ctx.save_for_backward(x)

        if inplace:
            q_output = x
        else:
            q_output = x.clone()
        input_scaled = round_tensor(x * scale, round_mode)
        q_output.copy_(torch.clamp(input_scaled, min_val, max_val) / scale)

        # grad_scale - to mask the clipped values
        if min_val < 0:
            grad_scale = (input_scaled >= min_val) & (input_scaled <= max_val)
        else:
            grad_scale = (input_scaled >= 0) & (input_scaled <= max_val)

        ctx.grad_scale = grad_scale.to(x.dtype)

        return q_output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_scale = ctx.grad_scale
        grad_input = grad_output * grad_scale
        return grad_input, None, None, None, None, None


class FixedPointQuantizer:
    def __init__(self, bitwidth, frac_w=None, frac_b=None, frac_out=None, inplace=False):
        self.bitwidth = bitwidth
        self.frac_in = None
        self.frac_w = frac_w
        self.frac_b = frac_b
        self.frac_out = frac_out
        self.max_val = (1 << (bitwidth - 1)) - 1  # Max value based on bitwidth
        self.min_val = -self.max_val - 1  # Min value for signed fixed-point
        self.round_mode = 'HALF_UP'
        self.inplace = inplace

    def calc_frac_in(self, x):
        self.frac_in = find_fix_pos(x, self.bitwidth, 1, 2)
        return self.frac_in

    @property
    def get_frac_in(self):
        return self.frac_in

    @property
    def get_frac_w(self):
        return self.frac_w

    @property
    def get_frac_b(self):
        return self.frac_b

    @property
    def get_frac_out(self):
        return self.frac_out

    def get_weight_quantizer(self, w_type):
        quantizer_map = {
            'weight': self.quantize_weigth,
            'bias': self.quantize_bias,
            'out': self.quantize_out,
        }
        if w_type in quantizer_map:
            return quantizer_map[w_type]
        else:
            raise ValueError(f"Unknown weight type: {w_type}")

    def determine_frac(self, w_tensor):
        return find_fix_pos(w_tensor, self.bitwidth, 3, 2)

    def _fake_quantize(self, x, scale):
        return FakeQuantize.apply(x, scale, self.min_val, self.max_val, self.round_mode, self.inplace)

    def quantize_weigth(self, x):
        self.frac_w = self.determine_frac(x)
        scale = 2 ** self.frac_w  # Fixed-point scaling
        qout = self._fake_quantize(x, scale)
        return qout

    def quantize_bias(self, x):
        self.frac_b = self.determine_frac(x)
        scale = 2 ** self.frac_b  # Fixed-point scaling
        qout = self._fake_quantize(x, scale)
        return qout

    def quantize_out(self, x):
        self.frac_out = self.determine_frac(x)
        scale = 2 ** self.frac_out  # Fixed-point scaling
        qout = self._fake_quantize(x, scale)
        return qout

    def quantize(self, x, frac):
        scale = 2 ** frac  # Fixed-point scaling
        qout = self._fake_quantize(x, scale)
        return qout


def QuantStubF(x):
    quantizer = FixedPointQuantizer(bitwidth=8)
    input_quantizer = quantizer.get_weight_quantizer('out')
    return input_quantizer(x)


if __name__ == '__main__':
    quantizer = FixedPointQuantizer(bitwidth=8)
    w_quantizer = quantizer.get_weight_quantizer('weight')
    qw = w_quantizer(torch.tensor([0.4847, 0.4672]))
    print(type(quantizer.get_frac_b))
    print(quantizer.get_frac_w)

    print(qw)
    print(QuantStubF(torch.tensor([0.4847, 0.4672])))
