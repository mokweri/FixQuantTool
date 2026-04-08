import os
import argparse
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import helper, TensorProto

from fixquant.quantization.fix_ops import to_int_tensor, to_float_tensor
from fixquant.emulation.fxp_emu_modules import HLSConv2d, FXPConv2dTorch, HLSConv2dInt


# ----------------------------- Logging ---------------------------------
logger = logging.getLogger("hw_fxp_test")


# ----------------------------- ONNX helpers ----------------------------
def attach_attrs(node, attrs: Dict[str, int | float]):
    """Add / replace integer attributes on a node."""
    to_remove = [a for a in node.attribute if a.name in attrs]
    for a in to_remove:
        node.attribute.remove(a)
    for k, v in attrs.items():
        node.attribute.extend([helper.make_attribute(k, int(v))])


def export_conv_to_onnx(conv: nn.Conv2d,
                        frac_din: int, frac_w: int, frac_b: int, frac_out: int,
                        onnx_path: str = "conv_with_frac.onnx",
                        opset: int = 13,
                        store_int8_wb: bool = True) -> None:
    """Export a Conv2d to ONNX, optionally replacing initializers with raw INT8 and adding frac_* attrs."""
    dummy = torch.zeros(1, conv.in_channels, 224, 224)
    torch.onnx.export(conv, dummy, onnx_path,
                      export_params=True, opset_version=opset,
                      do_constant_folding=True,
                      input_names=["input"], output_names=["output"])

    model = onnx.load(onnx_path)

    if store_int8_wb:
        # Replace initializers with INT8 raw bytes
        W_init, B_init = model.graph.initializer[:2]
        # Infer suitable frac from args
        w_int8 = to_int_tensor(conv.weight, signed=True, n_bits=8, n_frac=frac_w)
        b_int8 = to_int_tensor(conv.bias,   signed=True, n_bits=8, n_frac=frac_b)

        W_init.data_type = TensorProto.INT8
        W_init.raw_data = w_int8.cpu().numpy().tobytes()
        B_init.data_type = TensorProto.INT8
        B_init.raw_data = b_int8.cpu().numpy().tobytes()

    # Add radix attributes to Conv nodes
    fp_attrs = dict(frac_input=frac_din, frac_W=frac_w, frac_B=frac_b, frac_output=frac_out)
    for n in model.graph.node:
        if n.op_type == "Conv":
            attach_attrs(n, fp_attrs)

    onnx.save(model, onnx_path)
    logger.info("ONNX saved → %s", onnx_path)


# ----------------------------- IO helpers ------------------------------

def read_quantized_parameters(
        weights_filepath: str,
        bias_filepath: str,
        weights_shape: Tuple[int, ...],
        bias_shape: Optional[Tuple[int, ...]] = None,
        dtype: np.dtype = np.int8
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Reads quantized weights and biases from binary files and reconstructs them into PyTorch tensors.
    Returns a pair (weights_tensor, bias_tensor or None).
    """

    def _read_and_reshape(filepath: str, shape: Optional[Tuple[int, ...]]) -> Optional[torch.Tensor]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file was not found: {filepath}")
        if shape is None or os.path.getsize(filepath) == 0:
            return None
        data_flat = np.fromfile(filepath, dtype=dtype)
        expected_elements = int(np.prod(shape))
        if data_flat.size != expected_elements:
            raise ValueError(
                f"Shape mismatch in {filepath}. Expected {expected_elements} for {shape}, got {data_flat.size}.")
        return torch.from_numpy(data_flat.reshape(shape))

    weights_tensor = _read_and_reshape(weights_filepath, weights_shape)
    bias_tensor = _read_and_reshape(bias_filepath, bias_shape)
    if weights_tensor is None:
        raise ValueError(f"Weight file '{weights_filepath}' is empty or could not be read.")
    return weights_tensor, bias_tensor


def load_qparams(json_path: Optional[str]) -> Dict[str, Any]:
    if json_path is None:
        return {}
    import json
    with open(json_path, "r") as f:
        return json.load(f)


# ----------------------------- Core ops --------------------------------
@torch.no_grad()
def build_fp_conv_from_int8(w_int8: torch.Tensor, b_int8: Optional[torch.Tensor],
                            frac_w: int, frac_b: int,
                            stride: Tuple[int, int], padding: Tuple[int, int],
                            dilation: Tuple[int, int] = (1, 1), groups: int = 1) -> nn.Conv2d:
    """Recreate a Conv2d in float32 from int8 weights/bias + frac params."""
    out_ch, in_ch, kH, kW = w_int8.shape
    conv = nn.Conv2d(in_ch, out_ch, (kH, kW), stride=stride, padding=padding,
                     dilation=dilation, groups=groups, bias=b_int8 is not None)
    conv.weight.data.copy_(to_float_tensor(w_int8, frac_w))
    if b_int8 is not None:
        conv.bias.data.copy_(to_float_tensor(b_int8, frac_b))
    conv.eval()
    return conv


def run_emulation(x_i8: torch.Tensor,
                  w_i8: torch.Tensor, b_i8: Optional[torch.Tensor],
                  stride: Tuple[int, int], padding: Tuple[int, int],
                  frac_in: int, frac_w: int, frac_b: int, frac_out: int,
                  relu: bool = False,
                  which: str = "hls") -> torch.Tensor:
    """Run int8 emulation. which∈{"hls","fxp","int"}. Returns int8 output tensor (NCHW)."""
    if which == "hls":
        emu = HLSConv2d(w_i8, b_i8,
                        stride=stride[0] if stride[0] == stride[1] else stride,
                        padding=padding[0] if padding[0] == padding[1] else padding,
                        frac_din=frac_in, frac_w=frac_w, frac_b=frac_b, frac_dout=frac_out,
                        relu=relu)
        y_i8 = emu(x_i8)
    elif which == "fxp":
        emu = FXPConv2dTorch(w_i8, b_i8,
                             stride=stride[0] if stride[0] == stride[1] else stride,
                             padding=padding[0] if padding[0] == padding[1] else padding,
                             frac_din=frac_in, frac_w=frac_w, frac_b=frac_b, frac_out=frac_out,
                             relu=relu)
        y_i8 = emu(x_i8)
    elif which == "int":
        emu = HLSConv2dInt(w_i8, b_i8,
                           stride=stride[0] if stride[0] == stride[1] else stride,
                           padding=padding[0] if padding[0] == padding[1] else padding,
                           relu=relu)
        y_i8 = emu(x_i8)
    else:
        raise ValueError("which must be one of {'hls','fxp','int'}")
    return y_i8


def compare_with_float_ref(x_i8: torch.Tensor, frac_in: int,
                           conv_fp: nn.Conv2d, frac_out: int,
                           relu: bool = False) -> Tuple[torch.Tensor, Dict[str, int | float]]:
    """Create a float ref from int8 input and conv_fp, quantize to int8 with frac_out, optionally ReLU, and compare."""
    x_fp = to_float_tensor(x_i8, frac_in)
    y_fp = conv_fp(x_fp)
    if relu:
        y_fp = torch.clamp_min(y_fp, 0.)
    y_ref_i8 = to_int_tensor(y_fp, signed=True, n_bits=8, n_frac=frac_out)

    stats = {}
    # If a separate emu output is provided by caller, they can compute diff. Here we return only ref and stats placeholder.
    stats["y_ref_min"] = int(y_ref_i8.min().item())
    stats["y_ref_max"] = int(y_ref_i8.max().item())
    return y_ref_i8, stats


def compute_diff_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, int | float]:
    """Compute |a-b| stats for int tensors (expects same shape)."""
    diff = (a.to(torch.int16) - b.to(torch.int16)).abs()
    return {
        "max_abs": int(diff.max().item()),
        "non_zero": int((diff != 0).sum().item()),
        "numel": int(diff.numel()),
    }


# ----------------------------- CLI -------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="INT8 conv emulation/validation using saved blobs + qparams")
    p.add_argument("--qparams_json", default=None,
                   help="Path to JSON with frac_* and shapes; if omitted, defaults under repo hw_data_files.")
    p.add_argument("--weights_file", default=None)
    p.add_argument("--bias_file", default=None)
    p.add_argument("--activation_file", default=None)
    p.add_argument("--out_file", default=None, help="Where to save the emulation int8 output")
    # Overrides (optional)
    p.add_argument("--frac_in", type=int)
    p.add_argument("--frac_w", type=int)
    p.add_argument("--frac_b", type=int)
    p.add_argument("--frac_out", type=int)
    p.add_argument("--weights_dim", nargs="+", type=int, help="OIHW")
    p.add_argument("--activation_dims", nargs="+", type=int, help="CHW or NCHW")
    p.add_argument("--stride", nargs="+", type=int)
    p.add_argument("--padding", nargs="+", type=int)
    # Emulation/ref options
    p.add_argument("--emu", choices=["hls", "fxp", "int"], default="hls")
    p.add_argument("--relu", action="store_true")
    p.add_argument("--compare_float_ref", action="store_true",
                   help="Also compute float reference and report diff stats")
    p.add_argument("--export_onnx", action="store_true")
    p.add_argument("--log", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))

    # Resolve repository root and default artifact paths
    REPO_ROOT = Path(__file__).resolve().parents[1]
    qparams_path = Path(args.qparams_json) if args.qparams_json else (REPO_ROOT / "outputs/hw_data_files/layer1_0_conv2_qparams.json")
    weights_path = Path(args.weights_file) if args.weights_file else (REPO_ROOT / "outputs/hw_data_files/weights_64x64x3x3.data")
    bias_path = Path(args.bias_file) if args.bias_file else (REPO_ROOT / "outputs/hw_data_files/biases_1x64.data")
    activation_path = Path(args.activation_file) if args.activation_file else (REPO_ROOT / "outputs/hw_data_files/test_image_64x64x64.data")
    out_path = Path(args.out_file) if args.out_file else (REPO_ROOT / "outputs/hw_data_files/t_ref_output.data")

    meta = load_qparams(str(qparams_path)) if qparams_path.exists() else {}

    # Resolve shapes/params from JSON with CLI overrides
    act_dims_raw = tuple(args.activation_dims) if args.activation_dims else tuple(meta.get("activation_in_dims", []))
    if not act_dims_raw:
        raise ValueError("Activation dims not provided; pass --activation_dims or supply in qparams JSON")

    # Normalize to NCHW (with batch=1) for emulation
    if len(act_dims_raw) == 3:
        in_shape_nchw = (1, act_dims_raw[0], act_dims_raw[1], act_dims_raw[2])
    elif len(act_dims_raw) == 4:
        in_shape_nchw = act_dims_raw
    else:
        in_shape_nchw = (1,) + act_dims_raw[-3:]

    wdim = tuple(args.weights_dim) if args.weights_dim else tuple(meta.get("weights_dim", []))
    if not wdim:
        raise ValueError("Weights dim not provided; pass --weights_dim or supply in qparams JSON")

    conv_params = meta.get("conv_params", {})
    pad = tuple(args.padding) if args.padding else tuple(conv_params.get("padding", (0, 0)))
    stride = tuple(args.stride) if args.stride else tuple(conv_params.get("stride", (1, 1)))

    # Fractions
    frac_in = args.frac_in if args.frac_in is not None else int(meta.get("frac_in", 5))
    frac_w = args.frac_w if args.frac_w is not None else int(meta.get("frac_w", 0))
    frac_b = args.frac_b if args.frac_b is not None else int(meta.get("frac_b", 0))
    frac_out = args.frac_out if args.frac_out is not None else int(meta.get("frac_out", 0))

    logger.info("Artifacts:\n  qparams=%s\n  weights=%s\n  bias=%s\n  activation=%s\n  out=%s", qparams_path, weights_path, bias_path, activation_path, out_path)
    logger.info("Using dims: act_shape=%s, W=%s, stride=%s, padding=%s", in_shape_nchw, wdim, stride, pad)
    logger.info("Using fracs: in=%d, w=%d, b=%d, out=%d", frac_in, frac_w, frac_b, frac_out)

    # Load int8 blobs
    w_i8, b_i8 = read_quantized_parameters(str(weights_path), str(bias_path), wdim, (wdim[0],))

    a_i8_np = np.fromfile(str(activation_path), dtype=np.int8)
    expected_a = int(np.prod(in_shape_nchw[1:]))
    if a_i8_np.size != expected_a:
        raise ValueError(f"Activation size mismatch: file has {a_i8_np.size}, expected {expected_a} for {in_shape_nchw[1:]}")
    x_i8 = torch.from_numpy(a_i8_np.reshape(in_shape_nchw))

    # Emulation
    y_i8 = run_emulation(x_i8, w_i8, b_i8, stride=stride, padding=pad,
                         frac_in=frac_in, frac_w=frac_w, frac_b=frac_b, frac_out=frac_out,
                         relu=args.relu, which=args.emu)

    # Save output
    torch.as_tensor(y_i8).cpu().numpy().astype("int8").tofile(str(out_path))
    logger.info("Saved emu output → %s  shape=%s", out_path, tuple(y_i8.shape))

    # Optional float reference & diff
    if args.compare_float_ref:
        conv_fp = build_fp_conv_from_int8(w_i8, b_i8, frac_w=frac_w, frac_b=frac_b, stride=stride, padding=pad)
        y_ref_i8, ref_stats = compare_with_float_ref(x_i8, frac_in=frac_in, conv_fp=conv_fp, frac_out=frac_out,
                                                     relu=args.relu)
        diffs = compute_diff_stats(y_i8, y_ref_i8)
        logger.info("Ref stats: %s", ref_stats)
        logger.info("Diff stats: %s", diffs)

    if args.export_onnx:
        conv_fp = build_fp_conv_from_int8(w_i8, b_i8, frac_w=frac_w, frac_b=frac_b, stride=stride, padding=pad)
        export_conv_to_onnx(conv_fp, frac_din=frac_in, frac_w=frac_w, frac_b=frac_b, frac_out=frac_out)


if __name__ == "__main__":
    main()
