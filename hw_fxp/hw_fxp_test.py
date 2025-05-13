import torch, onnx
from onnx import helper, numpy_helper, TensorProto

import torch.nn as nn
from pathlib import Path

from PIL import Image
import torchvision.transforms as transforms
from quantization.fix_ops import to_int_tensor, to_float_tensor

torch.backends.quantized.engine = "fbgemm"          # use 'qnnpack' on ARM/Apple

class FXPConv2dTorch(nn.Module):
    """
    INT8-in / INT8-out convolution that re-uses PyTorch’s INT8 kernel.
    It maps PipeCNN’s fixed-point “frac_*” parameters to
        scale = 2^(–frac) ,  zero_point = 128  (for activations)
    """

    def __init__(self,
                 weight_int8, bias_int8=None,
                 stride=1, padding=1, dilation=1, groups=1,
                 frac_din=7, frac_w=7, frac_b=7, frac_out=7,
                 relu=True):
        super().__init__()

        # ---------------- scales & zero-points --------------------------
        self.scale_in   = 2.0 ** (-frac_din)
        self.scale_w    = 2.0 ** (-frac_w)
        self.scale_out  = 2.0 ** (-frac_out)
        self.z_act  = 128                 # QUInt8 requires 0-255
        self.z_wt   = 0                   # QInt8 weight
        self.z_out  = 128

        # ---------------- store raw INT8 weight ------------------------
        self.register_buffer("w_int8", weight_int8.contiguous())

        w_q = torch.quantize_per_tensor( weight_int8.float() * self.scale_w,
                                         scale=self.scale_w, zero_point=self.z_wt, dtype=torch.qint8)

        # ---------------- bias (INT8 → fp32 once) ----------------------
        if bias_int8 is None:
            bias_int8 = torch.zeros(weight_int8.size(0), dtype=torch.int8)
        self.register_buffer("bias_int8", bias_int8.contiguous())
        bias_fp32 = bias_int8.float() * (2.0 ** (-frac_b))

        # --------------- pack weight + bias (+params) ------------------
        stride   = stride   if isinstance(stride,   tuple) else (stride,   stride)
        padding  = padding  if isinstance(padding,  tuple) else (padding,  padding)
        dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        self.packed_w = torch.ops.quantized.conv2d_prepack(w_q, bias_fp32, stride, padding, dilation, groups)

        # misc
        self.relu = relu

    # ------------------------------------------------------------------
    def forward(self, x_int8: torch.Tensor) -> torch.Tensor:
        """
        x_int8 : raw signed INT8 interpreted as real = v * 2^(–frac_din)
        returns: signed INT8 in Qm.frac_out
        """
        assert x_int8.dtype == torch.int8, "input must be raw INT8"

        # ---------- 1.   INT8  →  QUInt8 wrapper (shift by +128) -------
        x_q = torch.quantize_per_tensor(
                x_int8.float() * self.scale_in,
                scale=self.scale_in, zero_point=self.z_act, dtype=torch.quint8)

        # ---------- 2.   convolution  ----------------------------------
        y_q = torch.ops.quantized.conv2d(
                x_q, self.packed_w, self.scale_out, self.z_out)

        # ---------- 3.   back to signed INT8 (+ optional ReLU) ---------
        y_u8  = y_q.int_repr()                                   # uint8
        y_i16 = y_u8.to(torch.int16) - self.z_out                # centre at 0
        y_i8  = y_i16.to(torch.int8)

        if self.relu:
            y_i8 = torch.clamp_min(y_i8, 0)

        return y_i8


# ----------------------------------------------------------------------
def build_fp_layer(w_int8, b_int8,
                   stride=2, padding=3, dilation=1, groups=1,
                   frac_w=7, frac_b=7):
    """
    Returns an nn.Conv2d whose weights/biases are the *de-quantised*
    versions of the INT8 parameters that you feed into FXPConv2dTorch.
    """
    out_ch, in_ch, kH, kW = w_int8.shape
    conv_fp = nn.Conv2d(in_ch, out_ch,
                        kernel_size=(kH, kW),
                        stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=True)

    # INT8  ->  fp32
    w_fp32 = w_int8.float() * (2.0 ** (-frac_w))
    b_fp32 = b_int8.float() * (2.0 ** (-frac_b))

    conv_fp.weight.data.copy_(w_fp32)
    conv_fp.bias.data.copy_(b_fp32)
    conv_fp.eval()          # we never train it

    return conv_fp


def export_conv_to_onnx(conv_module,
                        frac_din, frac_w, frac_b, frac_out,
                        onnx_path   = "conv1.onnx",
                        opset       = 13,
                        int8_weight = False):

    dummy = torch.zeros(1, conv_module.in_channels, 224, 224)  # size only
    torch.onnx.export(conv_module, dummy,
                      onnx_path,
                      export_params=True,
                      opset_version=opset,
                      do_constant_folding=True)

    model = onnx.load(onnx_path)

    # --------------------------------------------------------------
    #   2.a  (optional) replace fp32 weight with raw INT8 weight
    # --------------------------------------------------------------
    if int8_weight:
        # the first initializer is the weight tensor
        W_init = model.graph.initializer[0]
        # convert its raw_data bytes to int8
        W_int8 = conv_module.weight.detach().to(torch.int8).cpu().numpy()
        W_init.data_type = TensorProto.INT8
        W_init.raw_data  = W_int8.tobytes()

    # --------------------------------------------------------------
    #   2.b  embed fixed-point meta-data -- @TODO I need attributes not metadata
    # --------------------------------------------------------------
    meta = dict(frac_din = str(frac_din),
                frac_w   = str(frac_w),
                frac_b   = str(frac_b),
                frac_out = str(frac_out))

    # remove a key if it already exists (avoids duplicates)
    existing = {p.key: p for p in model.metadata_props}
    for k, v in meta.items():
        if k in existing:
            existing[k].value = v
        else:
            entry = model.metadata_props.add()
            entry.key = k
            entry.value = v

    onnx.save(model, onnx_path)
    print(f"Saved ONNX with meta-data → {onnx_path}")

# ----------------------------------------------------------------------
# Smoke-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        return tensor


    @torch.no_grad()
    def load_layer_params(layer: nn.Module, path: str | Path,
                          device: torch.device | str | None = None):
        chkpt = torch.load(path, map_location="cpu")
        dev = device or layer.weight.device

        layer.weight.copy_(chkpt["weight_int8"].to(dev))

        if layer.bias is not None and "bias_int8" in chkpt:
            layer.bias.copy_(chkpt["bias_int8"].to(dev))


    test_image = preprocess_image("new.JPEG")
    test_image = to_int_tensor(test_image, signed=True, n_bits=8, n_frac=5)

    #----------------
    chkpt = torch.load("conv1.pth", map_location="cpu")
    w_int8 = chkpt["weight_int8"]
    b_int8 = chkpt["bias_int8"]
    frac_w = chkpt["frac_w"]
    frac_b = chkpt["frac_b"]
    frac_out = chkpt["frac_out"]


    fxpconv = FXPConv2dTorch(w_int8, b_int8,
                             stride=2, padding=3,
                             frac_din=5, frac_w=frac_w, frac_b=frac_b, frac_out=frac_out,
                             relu=True)

    y_int8 = fxpconv(test_image)
    print("input shape:", test_image.shape, "dtype:", test_image.dtype)
    print("output shape:", y_int8.shape, "dtype:", y_int8.dtype)
    print("range:", y_int8.min().item(), y_int8.max().item())

    # ----------------------------------------------------------------------
    conv_fp = build_fp_layer(w_int8, b_int8,
                             stride=2, padding=3,
                             frac_w=frac_w, frac_b=frac_b)
    x_fp32 = to_float_tensor(test_image, n_frac=5)  # INT8 -> fp32
    y_fp32 = conv_fp(x_fp32)
    y_fp32 = torch.clamp_min(y_fp32, 0.) # relu
    # convert fp32 result to the *same* Qm.frac_out format
    y_ref_int8 = to_int_tensor(y_fp32, signed=True,
                               n_bits=8, n_frac=frac_out)

    diff = (y_int8.to(torch.int16) - y_ref_int8.to(torch.int16)).abs()

    print("------------------------------------------------------")
    print(f"INT8 kernel  output shape : {tuple(y_int8.shape)}")
    print(f"fp32→INT8 reference shape : {tuple(y_ref_int8.shape)}")
    print(f"max |Δ|              (LSB) : {diff.max().item()}")
    print(f"non-zero differences      : {(diff != 0).sum().item()}")
    print("------------------------------------------------------")

    # ---- export; set int8_weight=True if you prefer raw INT8 -------
    export_conv_to_onnx(conv_fp,
                        frac_din=5,
                        frac_w=frac_w,
                        frac_b=frac_b,
                        frac_out=frac_out,
                        onnx_path="conv1_with_frac.onnx",
                        int8_weight=False)  # or True