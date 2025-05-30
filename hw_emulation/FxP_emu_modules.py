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
        acc = torch.matmul(w_mat, patches)                     # INT32
        acc = acc.view(N, C_out, -1)                           # (N, C_out, L)

        # ------------------ 2.  Rounding & scaling -----------------------
        shift = self.fw + self.fin - self.fout                 # ≥0 is guaranteed by HLS host

        # HLS step: sign extension for right-shift rounding
        # In Python this is automatic, but we replicate bit logic for clarity
        acc_shifted = acc >> (shift - 1)                       # keep 1 extra bit
        # add first rounding bit ( +1 )
        acc_rnd1 = acc_shifted + 1

        # -------- saturation to 9 bits (MASK9B in the kernel) ------------
        acc_sat9 = sat_int(acc_rnd1, 9)                        # [-256, 255]

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
        out_h = (H + 2*self.padding - self.dilation*(kH - 1) - 1)//self.stride + 1
        out_w = (W + 2*self.padding - self.dilation*(kW - 1) - 1)//self.stride + 1
        return y_int8.view(N, C_out, out_h, out_w)
