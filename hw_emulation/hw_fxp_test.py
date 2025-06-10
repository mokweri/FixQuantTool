import os, torch, onnx, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper, TensorProto
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional

import torchvision.transforms as transforms
from quantization.fix_ops import to_int_tensor, to_float_tensor

from FxP_emu_modules import HLSConv2d, FXPConv2dTorch, HLSConv2dInt


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

def _subset_tensor(tensor: torch.Tensor, target_shape) -> torch.Tensor:
    """Helper function to subset a tensor based on target shape."""
    if target_shape is None:
        return tensor

    if len(target_shape) != tensor.ndim:
        raise ValueError(f"Target shape rank {len(target_shape)} mismatches tensor rank {tensor.ndim}")

    slicing_indices = []
    for i, dim_size in enumerate(target_shape):
        if not (0 < dim_size <= tensor.shape[i]):
            raise ValueError(f"Target dim {i} size {dim_size} exceeds original size {tensor.shape[i]}")
        slicing_indices.append(slice(0, dim_size))

    subset_tensor = tensor[tuple(slicing_indices)]
    print(f"Subsetted tensor from {tensor.shape} to {subset_tensor.shape}")
    return subset_tensor


def read_quantized_parameters(
        weights_filepath: str,
        bias_filepath: str,
        weights_shape: Tuple[int, ...],
        bias_shape: Optional[Tuple[int, ...]] = None,
        dtype: np.dtype = np.int8
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Reads quantized weights and biases from binary files and reconstructs them into PyTorch tensors.

    Args:
        weights_filepath (str): The file path for the quantized weights.
        bias_filepath (str): The file path for the quantized biases.
        weights_shape (Tuple[int, ...]): The target shape to reconstruct the weights tensor.
        bias_shape (Optional[Tuple[int, ...]], optional): The target shape for the bias tensor.
                                                          If the layer has no bias, this can be None.
                                                          Defaults to None.
        dtype (np.dtype, optional): The numpy data type of the stored parameters.
                                    Defaults to np.int8.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the reconstructed
        weights tensor and the bias tensor. The bias tensor is None if its file is empty
        or bias_shape is not provided.

    Raises:
        ValueError: If the number of elements in a file does not match the
                    number of elements expected by its corresponding shape.
        FileNotFoundError: If a specified file path does not exist.
    """

    def _read_and_reshape(filepath: str, shape: Optional[Tuple[int, ...]]) -> Optional[torch.Tensor]:
        """Helper to read a single binary file and reshape it."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file was not found: {filepath}")

        # Handle layers with no parameters (e.g., no bias) where an empty file is created.
        if shape is None or os.path.getsize(filepath) == 0:
            return None

        # Read the raw binary data from the file
        data_flat = np.fromfile(filepath, dtype=dtype)

        # Verify that the number of elements read matches the expected shape
        expected_elements = np.prod(shape)
        if data_flat.size != expected_elements:
            raise ValueError(
                f"Shape mismatch in {filepath}. "
                f"Expected {expected_elements} elements for shape {shape}, "
                f"but file contained {data_flat.size} elements."
            )

        # Reshape the numpy array and convert to a PyTorch tensor
        reconstructed_tensor = torch.from_numpy(data_flat.reshape(shape))
        return reconstructed_tensor

    # Read and reconstruct weights and biases using the helper
    weights_tensor = _read_and_reshape(weights_filepath, weights_shape)
    bias_tensor = _read_and_reshape(bias_filepath, bias_shape)

    # Ensure a tensor is returned for weights, as layers always have them.
    if weights_tensor is None:
        raise ValueError(f"Weight file '{weights_filepath}' is empty or could not be read.")

    return weights_tensor, bias_tensor


if __name__ == "__main__":
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

    """---- Load Data/Weights ------------------------"""
    # -----conv data from a full layer
    # w_int8, b_int8, frac_w, frac_b, frac_out = load_layer_params("conv1.pth")
    # STRIDE =2
    # PADDING=3

    # -----conv data from a saved small layer
    w_file = "../hw_data_files/weights_8x3x3x3.data"
    b_file = "../hw_data_files/biases_1x8.data"
    frac_w = 5
    frac_b=2
    frac_out=1
    STRIDE = 1
    PADDING = 1

    w_shape = (8, 3, 3, 3)
    b_shape = (8)
    w_int8, b_int8 = read_quantized_parameters(
        weights_filepath=w_file,
        bias_filepath=b_file,
        weights_shape=w_shape,
        bias_shape=b_shape,
        dtype=np.int8
    )


    fxp_conv = FXPConv2dTorch(w_int8, b_int8,
                              stride=STRIDE, padding=PADDING,
                              frac_din=5, frac_w=frac_w, frac_b=frac_b, frac_out=frac_out,
                              relu=True)

    hls_conv = HLSConv2d(w_int8, b_int8,
                         stride=STRIDE, padding=PADDING,
                         frac_din=5, frac_w=frac_w, frac_b=frac_b, frac_dout=frac_out,
                         relu=True)

    conv_int = HLSConv2dInt(w_int8, b_int8, stride=STRIDE, padding=PADDING,relu=True)

    """---- Input Image ------------------------"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    img = Image.open("new.JPEG").convert("RGB")
    x_fp  = transform(img).unsqueeze(0)
    x_i8  = to_int_tensor(x_fp, signed=True, n_bits=8, n_frac=5)
    # subset input..
    print(x_i8.shape)
    sm_input_shape = (1,3, 64,64)  # (N,in_channels, H, W)
    x_i8 = _subset_tensor(x_i8, sm_input_shape)
    print(x_i8.shape)

    """---- INT Convd Run ------------------------"""
    # y_i8 = fxp_conv(x_i8)
    y_i8 = hls_conv(x_i8)
    #y_i32 = conv_int(x_i8)

    print(f"Output shape: {y_i8.shape}")

    """---- Float CONVD Reference ------------------------"""
    # fp_conv = build_fp_conv(w_int8, b_int8, stride=2, padding=3, frac_w=frac_w, frac_b=frac_b)
    # y_ref_i8 = to_int_tensor(torch.clamp_min(fp_conv(to_float_tensor(x_i8, 5)), 0.),
    #                          signed=True, n_bits=8, n_frac=frac_out)
    #
    # diff = (y_i8.to(torch.int16) - y_ref_i8.to(torch.int16)).abs()
    # print("max |Δ| :", diff.max().item(),
    #       "  non-zero :", (diff != 0).sum().item())

    """---- ONNX Export ------------------------"""
    # export_conv_to_onnx(fp_conv,
    #                     frac_din=5, frac_w=frac_w,
    #                     frac_b=frac_b, frac_out=frac_out,
    #                     onnx_path="conv1_with_frac.onnx",
    #                     store_int8_wb=True)

    """---- Data files export ------------------------"""
    x_i8.numpy().astype("int8").tofile("t1_input.data")
    y_i8.numpy().astype("int8").tofile("t1_ref_output.data")
    w_int8.numpy().astype("int8").tofile("t1_weights.data")
    b_int8.numpy().astype("int8").tofile("t1_biases.data")
    # print(y_i32[0])
    # print("Binary blobs written.")