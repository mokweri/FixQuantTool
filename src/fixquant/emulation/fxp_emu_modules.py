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


class FXPConv2dTorch(nn.Module):
    """INT8-in / INT8-out Conv2d that re-uses PyTorch’s quantised kernel."""
    def __init__(self, weight_int8, bias_int8=None,
                 stride=1, padding=1, dilation=1, groups=1,
                 frac_din=7, frac_w=7, frac_b=7, frac_out=7,
                 relu=True):
        super().__init__()

        self.scale_in  = 2.0 ** (-frac_din)
        self.scale_w   = 2.0 ** (-frac_w)
        self.scale_out = 2.0 ** (-frac_out)
        self.z_act  = 128                     # QUInt8 [0…255]
        self.z_wt   = 0                       # QInt8
        self.z_out  = 128

        self.register_buffer("w_int8", weight_int8.contiguous())
        w_q = torch.quantize_per_tensor(weight_int8.float()*self.scale_w,
                                        scale=self.scale_w, zero_point=self.z_wt,
                                        dtype=torch.qint8)

        if bias_int8 is None:
            bias_int8 = torch.zeros(weight_int8.size(0), dtype=torch.int8)
        self.register_buffer("bias_int8", bias_int8.contiguous())
        bias_fp32 = bias_int8.float() * (2.0**(-frac_b))

        stride   = stride   if isinstance(stride,   tuple) else (stride,   stride)
        padding  = padding  if isinstance(padding,  tuple) else (padding,  padding)
        dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        self.packed_w = torch.ops.quantized.conv2d_prepack(
                            w_q, bias_fp32, stride, padding, dilation, groups)
        self.relu = relu

    def forward(self, x_int8: torch.Tensor) -> torch.Tensor:
        assert x_int8.dtype == torch.int8
        x_q = torch.quantize_per_tensor(x_int8.float()*self.scale_in,
                                        scale=self.scale_in, zero_point=self.z_act,
                                        dtype=torch.quint8)
        y_q = torch.ops.quantized.conv2d(x_q, self.packed_w,
                                         self.scale_out, self.z_out)
        y_i8 = (y_q.int_repr().to(torch.int16) - self.z_out).to(torch.int8)
        return torch.clamp_min(y_i8, 0) if self.relu else y_i8


class HLSConv2d(torch.nn.Module):
    """
    Bit-true emulation of PipeCNN conv_pipe.cl (integer part only).
    Parameters
    ----------
      weight_int8           : (C_out, C_in, kH, kW)  torch.int8
      bias_int8             : (C_out,)               torch.int8   (may pass None)
      stride, padding, dilation, groups     : as in nn.Conv2d
      frac_w, frac_b, frac_din, frac_dout   : number of fractional bits in each quantity
      relu                  :  if True reproduce the (contol & 0x01) ReLU in the kernel
    """
    def __init__(self, weight_int8, bias_int8,
                 stride=1, padding=0, dilation=1, groups=1,
                 frac_w=7, frac_b=7, frac_din=7, frac_dout=7,
                 relu=True):
        super().__init__()
        self.register_buffer('w_int8', weight_int8.clone().contiguous())
        self.register_buffer('b_int8',
                             torch.zeros(weight_int8.size(0), dtype=torch.int8) if bias_int8 is None
                             else bias_int8.clone().contiguous())

        self.stride, self.padding = stride, padding
        self.dilation, self.groups = dilation, groups

        # frac parameters are stored for later use
        self.fw, self.fb = frac_w, frac_b
        self.fin, self.fout = frac_din, frac_dout
        self.relu = relu

        # widen bias once (same as HLS: sign-extend to 32b)
        self.register_buffer('bias_i32',
            self.b_int8.to(torch.int32))                     # raw INT8 pattern in int32 vessel

    # ---------------------------------------------------------------------
    def forward(self, x_int8: torch.Tensor) -> torch.Tensor:
        """
        x_int8 : (N, C_in, H, W)  torch.int8     (Qm.fin)
        returns : torch.int8 in Qm.fout
        """
        assert x_int8.dtype == torch.int8
        N, C_in, H, W = x_int8.shape
        C_out, _, kH, kW = self.w_int8.shape

        # ------------------ 1.  int-32 convolution -----------------------
        patches = F.unfold(
            x_int8.float(),  # ←  float32 view
            (kH, kW),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        ).to(torch.int32)  # ←  convert back to int32

        # weights -> (C_out, C_in*kH*kW)
        w_mat = self.w_int8.view(C_out, -1).to(torch.int32)

        # gemm : (N, C_out, L)
        # int32 matmul is not supported on CUDA, so we use float64 (double) to avoid precision loss
        acc_fp64 = torch.matmul(w_mat.to(torch.float64), patches.to(torch.float64))
        acc = torch.round(acc_fp64).to(torch.int32)
        acc = acc.view(N, C_out, -1)                           # (N, C_out, L)

        # ------------------ 2.  Rounding & scaling -----------------------
        shift = self.fw + self.fin - self.fout                 # ≥0 is guaranteed by HLS host

        # HLS step: sign extension for right-shift rounding
        # In PyTorch, right shift by negative number returns 0, destroying the tensor!
        # If shift - 1 < 0, we must left shift instead.
        if (shift - 1) >= 0:
            acc_shifted = acc >> (shift - 1)
        else:
            acc_shifted = acc << (1 - shift)
            
        # add first rounding bit ( +1 )
        acc_rnd1 = acc_shifted + 1

        # We DO NOT saturate to 9 bits here because the QAT model does not simulate
        # intermediate clipping before bias addition!
        acc_sat9 = acc_rnd1

        # ------------------ 3.  add bias (in 2-step rounding) ------------
        if self.fb == self.fout:
            bias_adj = (self.bias_i32 << 1)                    # <<1 to align with rnd-bit
        elif self.fb > self.fout:
            bias_adj = self.bias_i32 >> (self.fb - self.fout - 1)
        else:  # fb < fout
            bias_adj = self.bias_i32 << (self.fout - self.fb + 1)

        # broadcast bias over spatial dimension
        bias_adj = bias_adj.view(1, C_out, 1)
        acc_bias = acc_sat9 + bias_adj + 1                     # second +1 for 2-step rnd

        # ------------------ 4.  final truncation to 8 bits ---------------
        y_int8 = (acc_bias >> 1)                               # discard last rnd-bit
        y_int8 = sat_int(y_int8, 8).to(torch.int8)             # clip to [-128,127]

        # ------------------ 5.  optional ReLU ----------------------------
        if self.relu:
            y_int8 = torch.where(y_int8 < 0,
                                 torch.zeros_like(y_int8, dtype=torch.int8),
                                 y_int8)

        # reshape back to (N, C_out, H_out, W_out)
        pad_h = self.padding[0] if isinstance(self.padding, tuple) else self.padding
        pad_w = self.padding[1] if isinstance(self.padding, tuple) else self.padding
        dil_h = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
        dil_w = self.dilation[1] if isinstance(self.dilation, tuple) else self.dilation
        str_h = self.stride[0] if isinstance(self.stride, tuple) else self.stride
        str_w = self.stride[1] if isinstance(self.stride, tuple) else self.stride
        
        out_h = (H + 2*pad_h - dil_h*(kH - 1) - 1)//str_h + 1
        out_w = (W + 2*pad_w - dil_w*(kW - 1) - 1)//str_w + 1
        return y_int8.view(N, C_out, out_h, out_w)

class HLSConv2dInt(torch.nn.Module):
    """
    INT ONLY Conv2d - NO fixed point quantization
    Parameters
    ----------
      weight_int8           : (C_out, C_in, kH, kW)  torch.int8
      bias_int8             : (C_out,)               torch.int8   (may pass None)
      stride, padding, dilation, groups   : as in nn.Conv2d
      relu                  :  if True do relu activation
    """
    def __init__(self, weight_int8, bias_int8,
                 stride=1, padding=0, dilation=1, groups=1,
                 relu=True):
        super().__init__()
        self.register_buffer('w_int8', weight_int8.clone().contiguous())
        self.register_buffer('b_int8', torch.zeros(weight_int8.size(0), dtype=torch.int8) if bias_int8 is None
                             else bias_int8.clone().contiguous())

        self.stride, self.padding = stride, padding
        self.dilation, self.groups = dilation, groups
        self.relu = relu

        # widen bias once (same as HLS: sign-extend to 32b)
        self.register_buffer('bias_i32',self.b_int8.to(torch.int32))   # raw INT8 pattern in int32 vessel

    # ---------------------------------------------------------------------
    def forward(self, x_int8: torch.Tensor) -> torch.Tensor:
        """
        x_int8 : (N, C_in, H, W)  torch.int8
        returns : torch.int32
        """
        assert x_int8.dtype == torch.int8
        N, C_in, H, W = x_int8.shape
        C_out, _, kH, kW = self.w_int8.shape

        # ------------------ 1.  int-32 convolution -----------------------
        patches = F.unfold(
            x_int8.float(),  # ←  float32 view
            (kH, kW),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        ).to(torch.int32)  # ←  convert back to int32

        # weights -> (C_out, C_in*kH*kW)
        w_mat = self.w_int8.view(C_out, -1).to(torch.int32)

        # gemm : (N, C_out, L)
        acc = torch.matmul(w_mat, patches)        # INT32
        acc = acc.view(N, C_out, -1)              # (N, C_out, L)

        # broadcast bias over spatial dimension
        bias = self.bias_i32.view(1, C_out, 1)
        y_int32 = acc + bias

        # ------------------ 5.  optional ReLU ----------------------------
        if self.relu:
            y_int32 = torch.where(y_int32 < 0,
                                 torch.zeros_like(y_int32, dtype=torch.int32),
                                 y_int32)

        # reshape back to (N, C_out, H_out, W_out)
        out_h = (H + 2*self.padding - self.dilation*(kH - 1) - 1)//self.stride + 1
        out_w = (W + 2*self.padding - self.dilation*(kW - 1) - 1)//self.stride + 1
        return y_int32.view(N, C_out, out_h, out_w)

class HLSLinear(torch.nn.Module):
    def __init__(self, weight_int8, bias_int8, frac_w=7, frac_b=7, frac_din=7, frac_dout=7):
        super().__init__()
        self.register_buffer('w_int8', weight_int8.clone().contiguous())
        self.register_buffer('b_int8',
                             torch.zeros(weight_int8.size(0), dtype=torch.int8) if bias_int8 is None
                             else bias_int8.clone().contiguous())
        self.fw, self.fb = frac_w, frac_b
        self.fin, self.fout = frac_din, frac_dout
        self.register_buffer('bias_i32', self.b_int8.to(torch.int32))

    def forward(self, x_int8: torch.Tensor) -> torch.Tensor:
        assert x_int8.dtype == torch.int8
        N = x_int8.size(0)
        x_flat = x_int8.view(N, -1).to(torch.float64)
        w_mat = self.w_int8.to(torch.float64)
        
        acc_fp64 = torch.matmul(x_flat, w_mat.t())
        acc = torch.round(acc_fp64).to(torch.int32)

        shift = self.fw + self.fin - self.fout
        if (shift - 1) >= 0:
            acc_shifted = acc >> (shift - 1)
        else:
            acc_shifted = acc << (1 - shift)
            
        acc_rnd1 = acc_shifted + 1
        acc_sat9 = acc_rnd1 # No intermediate saturation

        if self.fb == self.fout:
            bias_adj = (self.bias_i32 << 1)
        elif self.fb > self.fout:
            bias_adj = self.bias_i32 >> (self.fb - self.fout - 1)
        else:
            bias_adj = self.bias_i32 << (self.fout - self.fb + 1)

        bias_adj = bias_adj.view(1, -1)
        acc_bias = acc_sat9 + bias_adj + 1

        y_int8 = (acc_bias >> 1)
        y_int8 = sat_int(y_int8, 8).to(torch.int8)
        return y_int8

class HLSMaxPool2D(torch.nn.MaxPool2d):
    def forward(self, x_int8):
        assert x_int8.dtype == torch.int8
        x_f = x_int8.float()
        y_f = super().forward(x_f)
        return y_f.to(torch.int8)

class HLSAdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d):
    def forward(self, x_int8):
        assert x_int8.dtype == torch.int8
        x_f = x_int8.float()
        y_f = super().forward(x_f)
        return torch.round(y_f).to(torch.int8)

class HLSElementwiseAdd(torch.nn.Module):
    def __init__(self, frac_in1=5, frac_in2=5, frac_out=5):
        super().__init__()
        self.f1 = frac_in1
        self.f2 = frac_in2
        self.fout = frac_out
        
    def forward(self, x1, x2):
        assert x1.dtype == torch.int8 and x2.dtype == torch.int8
        y1 = x1.to(torch.int32)
        y2 = x2.to(torch.int32)
        
        # Align y1 to fout
        if self.f1 > self.fout:
            y1 = (y1 + (1 << (self.f1 - self.fout - 1))) >> (self.f1 - self.fout)
        elif self.f1 < self.fout:
            y1 = y1 << (self.fout - self.f1)
            
        # Align y2 to fout
        if self.f2 > self.fout:
            y2 = (y2 + (1 << (self.f2 - self.fout - 1))) >> (self.f2 - self.fout)
        elif self.f2 < self.fout:
            y2 = y2 << (self.fout - self.f2)
            
        y = y1 + y2
        return sat_int(y, 8).to(torch.int8)

class HLSRelu(torch.nn.ReLU):
    def forward(self, x_int8):
        assert x_int8.dtype == torch.int8
        return torch.clamp_min(x_int8, 0)

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


class TileCNNConv2d(torch.nn.Module):
    """
    Bit-exact digital-twin of the TileCNN conv2d hardware kernel.

    Reproduces the two-step rounding used in the HLS EMIT_LOOP:
        s1  = (acc >> (shift-1)) + 1
        out = (s1 + bias_adj + 1) >> 1
        out = clamp(out, -128, 127)
        out = relu(out)  [optional]

    Optional fused residual add (performed after conv+bias, before post_add_relu).

    Parameters
    ----------
    weight_int8   : (C_out, C_in, kH, kW)  torch.int8
    bias_int8     : (C_out,)               torch.int8  (pass None for zero bias)
    stride, padding, dilation, groups      : as in nn.Conv2d
    frac_w, frac_b, frac_din, frac_dout   : fractional widths
    relu          : apply ReLU after conv (but before residual add)
    residual_frac : fractional width of the residual tensor (needed if residual_add=True)
    residual_add  : fuse a residual add into this layer
    post_add_relu : apply ReLU after the residual add
    """
    def __init__(self, weight_int8, bias_int8,
                 stride=1, padding=0, dilation=1, groups=1,
                 frac_w=7, frac_b=7, frac_din=7, frac_dout=7,
                 relu=False,
                 residual_frac: int = 0,
                 residual_add: bool = False,
                 post_add_relu: bool = False):
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
        self.relu = relu
        self.residual_frac = residual_frac
        self.residual_add  = residual_add
        self.post_add_relu = post_add_relu

    # ------------------------------------------------------------------
    def _conv_int(self, x_int8: torch.Tensor) -> torch.Tensor:
        """Raw INT-32 convolution via float64 matmul (CUDA-safe)."""
        N, C_in, H, W = x_int8.shape
        C_out, _, kH, kW = self.w_int8.shape
        patches = F.unfold(
            x_int8.float(), (kH, kW),
            dilation=self.dilation, padding=self.padding, stride=self.stride
        ).to(torch.float64)                        # (N, C_in*kH*kW, L)
        w_mat = self.w_int8.view(C_out, -1).to(torch.float64)   # (C_out, C_in*kH*kW)
        acc = torch.round(torch.matmul(w_mat, patches)).to(torch.int64)  # (N, C_out, L)
        return acc.view(N, C_out, -1)

    # ------------------------------------------------------------------
    def forward(self, x_int8: torch.Tensor,
                residual: torch.Tensor = None) -> torch.Tensor:
        assert x_int8.dtype == torch.int8
        N, C_in, H, W = x_int8.shape
        _, _, kH, kW = self.w_int8.shape

        acc = self._conv_int(x_int8)               # (N, C_out, L)

        # --- Stage 1: two-step rounding + bias (TileCNN EMIT_LOOP) -------
        shift_out = self.fw + self.fin - self.fout
        if shift_out > 0:
            s1 = (acc >> (shift_out - 1)) + 1
        elif shift_out == 0:
            s1 = (acc << 1) + 1
        else:
            s1 = (acc << (1 - shift_out)) + 1

        bias_shift = self.fout - self.fb + 1
        bias_adj = _tc_bias_shift(self.bias_i64, bias_shift).view(1, -1, 1)
        out64 = (s1 + bias_adj + 1) >> 1          # (N, C_out, L)

        # --- Saturate to 8 bits -------------------------------------------
        out = torch.clamp(out64, -128, 127).to(torch.int8)

        # --- Optional standalone ReLU (no residual add yet) ---------------
        if self.relu and not self.residual_add:
            out = torch.clamp_min(out, 0)

        # --- Reshape to spatial map ---------------------------------------
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation
        str_h, str_w = self.stride
        out_h = (H + 2*pad_h - dil_h*(kH - 1) - 1)//str_h + 1
        out_w = (W + 2*pad_w - dil_w*(kW - 1) - 1)//str_w + 1
        C_out = self.w_int8.shape[0]
        out = out.view(N, C_out, out_h, out_w)

        # --- Fused residual add -------------------------------------------
        if self.residual_add:
            if residual is None:
                raise ValueError("TileCNNConv2d: residual_add=True but no residual tensor supplied")
            residual_shift = self.fout - self.residual_frac
            res_aligned = _tc_signed_shift(residual.to(torch.int64), residual_shift)
            summed = out.to(torch.int64) + res_aligned
            out = torch.clamp(summed, -128, 127).to(torch.int8)
            if self.post_add_relu:
                out = torch.clamp_min(out, 0)

        return out


class TileCNNLinear(torch.nn.Module):
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
            s1 = (acc >> (shift_out - 1)) + 1
        elif shift_out == 0:
            s1 = (acc << 1) + 1
        else:
            s1 = (acc << (1 - shift_out)) + 1

        bias_shift = self.fout - self.fb + 1
        bias_adj = _tc_bias_shift(self.bias_i64, bias_shift).view(1, -1)
        out64 = (s1 + bias_adj + 1) >> 1
        return torch.clamp(out64, -128, 127).to(torch.int8)


class TileCNNGAP(torch.nn.Module):
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


class TileCNNMaxPool(torch.nn.Module):
    """
    Bit-exact max-pool matching TileCNN hardware (3×3, stride=2, padding=1).
    The post_pool_shift is applied identically to _tilecnn_maxpool.
    """
    def __init__(self, kernel_size=3, stride=2, padding=1, post_pool_shift: int = 0):
        super().__init__()
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.post_pool_shift = post_pool_shift

    def forward(self, x_int8: torch.Tensor) -> torch.Tensor:
        assert x_int8.dtype == torch.int8
        y = F.max_pool2d(x_int8.float(),
                         kernel_size=self.kernel_size,
                         stride=self.stride,
                         padding=self.padding).to(torch.int64)
        out = _tc_signed_shift(y, self.post_pool_shift)
        return torch.clamp(out, -128, 127).to(torch.int8)
