import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.quantized.engine = "fbgemm"      # use 'qnnpack' on ARM/Apple
torch.manual_seed(0)


def sat_int(value, bits):
    """
    Clip signed integer tensor to the range representable by <bits> bits.
    Example: bits=9  ->  [-256, 255]
    """
    limit = 1 << (bits - 1)
    return torch.clamp(value, -limit, limit - 1)

def sign_extend(value, src_bits, dst_bits=32):
    """
    Extend <src_bits>-wide signed int to <dst_bits> bits.
    value is a torch.int32 tensor that already holds the raw pattern.
    """
    mask = (1 << src_bits) - 1
    value = value & mask                         # drop upper garbage
    sign  = 1 << (src_bits - 1)
    return (value ^ sign) - sign                 # textbook sign-extension


class HardwareElementwiseAdd(torch.nn.Module):
    def __init__(self, frac_in1=5, frac_in2=5, frac_out=5):
        super().__init__()
        self.f1 = frac_in1
        self.f2 = frac_in2
        self.fout = frac_out
        
    def forward(self, x1, x2):
        assert x1.dtype == torch.int8 and x2.dtype == torch.int8
        # Align both inputs to fout with the same rounding shift the TileCNN
        # residual path uses (_signed_shift), then add and saturate.
        y1 = _tc_signed_shift(x1, self.fout - self.f1)
        y2 = _tc_signed_shift(x2, self.fout - self.f2)
        y = y1 + y2
        return sat_int(y, 8).to(torch.int8)

class HardwareRelu(torch.nn.ReLU):
    def forward(self, x_int8):
        assert x_int8.dtype == torch.int8
        return torch.clamp_min(x_int8, 0)

class HardwareRelu6(torch.nn.Module):
    def __init__(self, frac_in):
        super().__init__()
        self.frac_in = frac_in

    def forward(self, x_int8):
        assert x_int8.dtype == torch.int8
        max_val = min(127, int(round(6.0 * (2 ** self.frac_in))))
        return torch.clamp(x_int8, min=0, max=max_val)

class InputQuantizer(torch.nn.Module):
    def __init__(self, frac_in):
        super().__init__()
        self.frac_in = frac_in
        
    def forward(self, x):
        from fixquant.quantization.fix_ops import to_int_tensor
        return to_int_tensor(x, signed=True, n_bits=8, n_frac=self.frac_in).to(torch.int8)

class OutputDequantizer(torch.nn.Module):
    def __init__(self, frac_out):
        super().__init__()
        self.frac_out = frac_out
        
    def forward(self, x_int8):
        assert x_int8.dtype == torch.int8
        return x_int8.float() / (2.0 ** self.frac_out)


# ===========================================================================
#  TileCNN Digital-Twin Modules
#  These implement the EXACT same fused integer arithmetic as
#  _tilecnn_conv2d / _tilecnn_linear / _tilecnn_gap in tilecnn_exporter.py.
#  Use them when you want a PyTorch model whose accuracy == FPGA accuracy.
# ===========================================================================

GAP_SCALE_FRAC_BITS = 16   # must match tilecnn_exporter.py constant


def _tc_bias_shift(values: torch.Tensor, shift: int) -> torch.Tensor:
    """Align bias to the (out_frac+1) scale used inside the TileCNN pipeline."""
    values = values.to(torch.int64)
    if shift >= 0:
        return values << shift
    return values >> (-shift)


def _tc_signed_shift(values: torch.Tensor, shift: int) -> torch.Tensor:
    """Rounding right-shift (or left-shift for negative shift) matching _signed_shift."""
    values = values.to(torch.int64)
    if shift > 0:
        return values << shift
    if shift < 0:
        s = -shift
        round_bias = (1 << (s - 1)) + (values >> 63)   # convergent towards -inf
        return (values + round_bias) >> s
    return values


class HardwareConv2d(torch.nn.Module):
    """
    Bit-exact digital-twin of the TileCNN conv2d hardware kernel.

    Reproduces the two-step round-half-up used in the HLS EMIT_LOOP:
        s1  = (acc >> (shift-1))
        out = (s1 + bias_adj + 1) >> 1
        out = clamp(out, -128, 127)

    Parameters
    ----------
    weight_int8   : (C_out, C_in, kH, kW)  torch.int8
    bias_int8     : (C_out,)               torch.int8  (pass None for zero bias)
    stride, padding, dilation, groups      : as in nn.Conv2d
    frac_w, frac_b, frac_din, frac_dout   : fractional widths
    """
    def __init__(self, weight_int8, bias_int8,
                 stride=1, padding=0, dilation=1, groups=1,
                 frac_w=7, frac_b=7, frac_din=7, frac_dout=7,
                 backend: str = 'tilecnn',
                 relu=False, relu6=False,
                 residual_frac=0, residual_add=False,
                 post_add_relu=False, post_add_relu6=False):
        super().__init__()
        self.register_buffer('w_int8', weight_int8.clone().contiguous())
        self.register_buffer('b_int8',
                             torch.zeros(weight_int8.size(0), dtype=torch.int8)
                             if bias_int8 is None else bias_int8.clone().contiguous())
        self.register_buffer('bias_i64', self.b_int8.to(torch.int64))

        self.stride   = stride   if isinstance(stride,   tuple) else (stride,   stride)
        self.padding  = padding  if isinstance(padding,  tuple) else (padding,  padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups   = groups

        self.fw, self.fb = frac_w, frac_b
        self.fin, self.fout = frac_din, frac_dout
        self.backend = backend
        
        self.relu = relu
        self.relu6 = relu6
        self.residual_frac = residual_frac
        self.residual_add = residual_add
        self.post_add_relu = post_add_relu
        self.post_add_relu6 = post_add_relu6

    # ------------------------------------------------------------------
    def _conv_int(self, x_int8: torch.Tensor) -> torch.Tensor:
        """Raw INT-32 convolution via float64 matmul (CUDA-safe)."""
        N, C_in, H, W = x_int8.shape
        C_out, _, kH, kW = self.w_int8.shape
        acc_fp64 = F.conv2d(
            x_int8.to(torch.float64),
            self.w_int8.to(torch.float64),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        acc = torch.round(acc_fp64).to(torch.int64).view(N, C_out, -1)
        return acc

    def _tc_signed_shift(self, values: torch.Tensor, shift: int) -> torch.Tensor:
        """Rounding right-shift (or left-shift for negative shift)."""
        values = values.to(torch.int64)
        if shift > 0:
            return values << shift
        if shift < 0:
            s = -shift
            round_bias = (1 << (s - 1)) + (values >> 63)
            return (values + round_bias) >> s
        return values

    # ------------------------------------------------------------------
    def forward(self, x_int8: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass doing the bit-exact computation.
        """
        assert x_int8.dtype == torch.int8
        squeeze = False
        if x_int8.dim() == 3:          # (C, H, W)  — from eval loop
            x_int8 = x_int8.unsqueeze(0)
            squeeze = True

        N, C_in, H, W = x_int8.shape
        C_out = self.w_int8.size(0)

        # 1) Int32 Convolution
        acc = self._conv_int(x_int8)   # (N, C_out, H_out * W_out)

        # 2) First shift to the (out_frac+1) scale — truncate only. The single
        #    round-half-up +1 is applied in the second shift below. (An extra +1
        #    here added a constant +0.5/layer output bias; removed 2026-07 to
        #    match the corrected HLS EMIT_LOOP.)
        shift_out = self.fw + self.fin - self.fout
        if shift_out > 0:
            s1 = (acc >> (shift_out - 1))
        elif shift_out == 0:
            s1 = (acc << 1)
        else:
            s1 = (acc << ((-shift_out) + 1))

        # 3) Bias Add & Second Shift
        bias_shift = self.fout - self.fb + 1
        if bias_shift >= 0:
            bias_adj = (self.bias_i64 << bias_shift).view(1, C_out, 1)
        else:
            bias_adj = (self.bias_i64 >> (-bias_shift)).view(1, C_out, 1)
            
        out64 = (s1 + bias_adj + 1) >> 1
        out = torch.clamp(out64, -128, 127).to(torch.int8)

        # --- Optional standalone ReLU / ReLU6 (no residual add yet) ---------------
        if not self.residual_add:
            if self.relu:
                out = torch.clamp_min(out, 0)
            elif self.relu6:
                max_val = min(127, int(round(6.0 * (2 ** self.fout))))
                out = torch.clamp(out, min=0, max=max_val)

        # Reshape back to spatial
        H_out = (H + 2 * self.padding[0] - self.dilation[0] * (self.w_int8.size(2) - 1) - 1) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.dilation[1] * (self.w_int8.size(3) - 1) - 1) // self.stride[1] + 1
        out = out.view(N, C_out, H_out, W_out)
        
        # --- Fused residual add -------------------------------------------
        if self.residual_add:
            if residual is None:
                raise ValueError("HardwareConv2d: residual_add=True but no residual tensor supplied")
            residual_shift = self.fout - self.residual_frac
            res_aligned = self._tc_signed_shift(residual.to(torch.int64), residual_shift)
            summed = out.to(torch.int64) + res_aligned
            out = torch.clamp(summed, -128, 127).to(torch.int8)
            
            if self.post_add_relu:
                out = torch.clamp_min(out, 0)
            elif self.post_add_relu6:
                max_val = min(127, int(round(6.0 * (2 ** self.fout))))
                out = torch.clamp(out, min=0, max=max_val)

        return out.squeeze(0) if squeeze else out


class HardwareLinear(torch.nn.Module):
    """
    Bit-exact digital-twin of the TileCNN fully-connected layer.
    Mirrors _tilecnn_linear() in tilecnn_exporter.py.
    """
    def __init__(self, weight_int8, bias_int8, frac_w=7, frac_b=7, frac_din=7, frac_dout=7):
        super().__init__()
        self.register_buffer('w_int8', weight_int8.clone().contiguous())
        self.register_buffer('b_int8',
                             torch.zeros(weight_int8.size(0), dtype=torch.int8)
                             if bias_int8 is None else bias_int8.clone().contiguous())
        self.register_buffer('bias_i64', self.b_int8.to(torch.int64))
        self.fw, self.fb = frac_w, frac_b
        self.fin, self.fout = frac_din, frac_dout

    def forward(self, x_int8: torch.Tensor) -> torch.Tensor:
        assert x_int8.dtype == torch.int8
        N = x_int8.size(0)
        x_flat = x_int8.view(N, -1).to(torch.float64)
        w_mat  = self.w_int8.to(torch.float64)
        acc = torch.round(torch.matmul(x_flat, w_mat.t())).to(torch.int64)  # (N, C_out)

        shift_out = self.fw + self.fin - self.fout
        if shift_out > 0:
            s1 = (acc >> (shift_out - 1))
        elif shift_out == 0:
            s1 = (acc << 1)
        else:
            s1 = (acc << ((-shift_out) + 1))

        bias_shift = self.fout - self.fb + 1
        bias_adj = _tc_bias_shift(self.bias_i64, bias_shift).view(1, -1)
        out64 = (s1 + bias_adj + 1) >> 1
        return torch.clamp(out64, -128, 127).to(torch.int8)


class HardwareGAP(torch.nn.Module):
    """
    Bit-exact digital-twin of the TileCNN Global Average Pooling kernel.
    Mirrors _tilecnn_gap() in tilecnn_exporter.py.
    Uses a fixed-point reciprocal with GAP_SCALE_FRAC_BITS precision.
    """
    def __init__(self, frac_in: int, frac_out: int):
        super().__init__()
        self.fin  = frac_in
        self.fout = frac_out

    def forward(self, x_int8: torch.Tensor) -> torch.Tensor:
        assert x_int8.dtype == torch.int8
        squeeze = False
        if x_int8.dim() == 3:          # (C, H, W)  — from eval loop
            x_int8 = x_int8.unsqueeze(0)
            squeeze = True

        N, C, H, W = x_int8.shape
        num_elems = H * W
        total_shift = GAP_SCALE_FRAC_BITS + (self.fout - self.fin)
        if total_shift < 0:
            raise ValueError("TileCNNGAP: negative total_shift not supported")
        gap_mul = ((1 << total_shift) + num_elems // 2) // num_elems

        sums = x_int8.to(torch.int64).reshape(N, C, -1).sum(dim=2)   # (N, C)
        scaled = (sums * gap_mul + (1 << (GAP_SCALE_FRAC_BITS - 1))) >> GAP_SCALE_FRAC_BITS
        out = torch.clamp(scaled, -128, 127).to(torch.int8).reshape(N, C, 1, 1)
        return out.squeeze(0) if squeeze else out


class HardwareAdaptiveAvgPool2d(torch.nn.Module):
    """
    General AdaptiveAvgPool2d in int8 for emulation purposes.
    (Note: TileCNN hardware typically only supports Global Average Pooling via gap2d).
    """
    def __init__(self, output_size, frac_in: int, frac_out: int):
        super().__init__()
        self.output_size = output_size
        self.fin = frac_in
        self.fout = frac_out

    def forward(self, x_int8: torch.Tensor) -> torch.Tensor:
        assert x_int8.dtype == torch.int8
        x_f = x_int8.float()
        y_f = torch.nn.functional.adaptive_avg_pool2d(x_f, self.output_size)
        return torch.round(y_f).to(torch.int8)


class HardwareMaxPool2d(torch.nn.Module):
    """
    Bit-exact max-pool matching TileCNN hardware (3×3, stride=2, padding=1).
    The post_pool_shift is applied identically to _tilecnn_maxpool.
    """
    def __init__(self, kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False, post_pool_shift: int = 0):
        super().__init__()
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        self.return_indices = return_indices
        self.ceil_mode      = ceil_mode
        self.post_pool_shift = post_pool_shift

    def forward(self, x_int8: torch.Tensor) -> torch.Tensor:
        assert x_int8.dtype == torch.int8
        y = F.max_pool2d(x_int8.float(),
                         kernel_size=self.kernel_size,
                         stride=self.stride,
                         padding=self.padding,
                         dilation=self.dilation,
                         ceil_mode=self.ceil_mode,
                         return_indices=self.return_indices)
        
        # If return_indices is true, y is a tuple (tensor, indices)
        if self.return_indices:
            pool_val, indices = y
            pool_val = pool_val.to(torch.int64)
            out = _tc_signed_shift(pool_val, self.post_pool_shift)
            return torch.clamp(out, -128, 127).to(torch.int8), indices
        else:
            y = y.to(torch.int64)
            out = _tc_signed_shift(y, self.post_pool_shift)
            return torch.clamp(out, -128, 127).to(torch.int8)
