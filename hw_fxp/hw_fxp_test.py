#!/usr/bin/env python3
"""
Creates all assets needed by the HW testing:
    • conv1_with_frac.onnx      (Conv layer + frac_* attributes)
    • test_image.data           (input tensor  INT8 binary)
    • ref_output.data           (golden output INT8 binary)
    • weights.data              (raw INT8 weights)
    • biases.data               (raw INT8 biases)
The script also performs a self-check: INT8 kernel vs. fp32 reference
(max |Δ| == 0 / 1 LSB).
"""
import torch, onnx, numpy as np
import torch.nn as nn
from onnx import helper, TensorProto
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from quantization.fix_ops import to_int_tensor, to_float_tensor

from hls_conv import HLSConv2d

torch.backends.quantized.engine = "fbgemm"      # use 'qnnpack' on ARM/Apple
torch.manual_seed(0)

# ======================================================================
#   Classes / functions
# ======================================================================
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


def build_fp_conv(w_int8, b_int8, stride=1, padding=1, dilation=1, groups=1,
                  frac_w=7, frac_b=7):
    """Re-creates the very same layer in fp32."""
    out_ch, in_ch, kH, kW = w_int8.shape
    conv = nn.Conv2d(in_ch, out_ch, (kH, kW),
                     stride=stride, padding=padding,
                     dilation=dilation, groups=groups, bias=True)
    conv.weight.data.copy_(w_int8.float() * 2.0**(-frac_w))
    conv.bias.data.copy_(b_int8.float() * 2.0**(-frac_b))
    conv.eval()
    return conv


def attach_attrs(node, attrs: dict[str, int | float]):
    """Add / replace integer attributes on a node."""
    to_remove = [a for a in node.attribute if a.name in attrs]
    for a in to_remove:
        node.attribute.remove(a)
    for k, v in attrs.items():
        node.attribute.extend([helper.make_attribute(k, int(v))])


# ======================================================================
#   1.  load INT8 parameters   (QModel checkpoint)
# ======================================================================
def load_layer_params(chkpt_path):
    ckpt = torch.load(chkpt_path, map_location="cpu")
    return (ckpt["weight_int8"], ckpt["bias_int8"],
            ckpt["frac_w"], ckpt["frac_b"], ckpt["frac_out"])


# ======================================================================
#   2.  export conv → ONNX   (with INT8 initialisers + radix attrs)
# ======================================================================
def export_conv_to_onnx(conv: nn.Conv2d,
                        frac_din, frac_w, frac_b, frac_out,
                        onnx_path       = "conv1_with_frac.onnx",
                        opset           = 13,
                        store_int8_wb   = True):

    dummy = torch.zeros(1, conv.in_channels, 224, 224)
    torch.onnx.export(conv, dummy, onnx_path,
                      export_params=True, opset_version=opset,
                      do_constant_folding=True,
                      input_names=["input"], output_names=["output"])

    model = onnx.load(onnx_path)

    # ---------- replace initialisers with RAW INT8 bytes ----------
    if store_int8_wb:
        W_init, B_init = model.graph.initializer[:2]
        w_int8 = to_int_tensor(conv.weight, signed=True, n_bits=8, n_frac=frac_w)
        b_int8 = to_int_tensor(conv.bias,   signed=True, n_bits=8, n_frac=frac_b)

        W_init.data_type = TensorProto.INT8
        W_init.raw_data  = w_int8.cpu().numpy().tobytes()
        B_init.data_type = TensorProto.INT8
        B_init.raw_data  = b_int8.cpu().numpy().tobytes()

    # ---------- add four radix attributes to every Conv -----------
    fp_attrs = dict(frac_input=frac_din, frac_W=frac_w, frac_B=frac_b, frac_output=frac_out)
    for n in model.graph.node:
        if n.op_type == "Conv":
            attach_attrs(n, fp_attrs)

    onnx.save(model, onnx_path)
    print(f"ONNX saved  →  {onnx_path}")


if __name__ == "__main__":

    # ----------------------------------------------------- load data
    w_int8, b_int8, frac_w, frac_b, frac_out = load_layer_params("conv1.pth")

    fxp_conv = FXPConv2dTorch(w_int8, b_int8,
                              stride=2, padding=3,
                              frac_din=5, frac_w=frac_w,
                              frac_b=frac_b, frac_out=frac_out,
                              relu=True)

    hls_conv = HLSConv2d(w_int8, b_int8,
                         stride=2, padding=3,
                         frac_din=5, frac_w=frac_w,
                         frac_b=frac_b, frac_dout=frac_out,
                         relu=True)

    # ----------------------------------------------------- test image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    img = Image.open("new.JPEG").convert("RGB")
    x_fp  = transform(img).unsqueeze(0)
    x_i8  = to_int_tensor(x_fp, signed=True, n_bits=8, n_frac=5)

    # ----------------------------------------------------- INT8 run
    # y_i8 = fxp_conv(x_i8)
    y_i8 = hls_conv(x_i8)

    # ----------------------------------------------------- fp32 reference
    fp_conv = build_fp_conv(w_int8, b_int8, stride=2, padding=3, frac_w=frac_w, frac_b=frac_b)
    y_ref_i8 = to_int_tensor(torch.clamp_min(fp_conv(to_float_tensor(x_i8, 5)), 0.),
                             signed=True, n_bits=8, n_frac=frac_out)
    print(y_i8)
    y_i8.numpy().astype("int8").tofile("ref_output2.data")

    diff = (y_i8.to(torch.int16) - y_ref_i8.to(torch.int16)).abs()
    print("max |Δ| :", diff.max().item(),
          "  non-zero :", (diff != 0).sum().item())

    # ----------------------------------------------------- export
    # export_conv_to_onnx(fp_conv,
    #                     frac_din=5, frac_w=frac_w,
    #                     frac_b=frac_b, frac_out=frac_out,
    #                     onnx_path="conv1_with_frac.onnx",
    #                     store_int8_wb=True)

    # ----------------------------------------------------- binaries
    # x_i8.numpy().astype("int8").tofile("test_image.data")
    # y_i8.numpy().astype("int8").tofile("ref_output.data")
    # w_int8.numpy().astype("int8").tofile("weights.data")
    # b_int8.numpy().astype("int8").tofile("biases.data")
    # print("Binary blobs written.")