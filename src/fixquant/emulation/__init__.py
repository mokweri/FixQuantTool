"""Hardware emulation modules for bit-exact FPGA-like inference."""

from .fxp_emu_modules import HardwareConv2d, HardwareLinear, HardwareMaxPool2d, HardwareGAP, HardwareAdaptiveAvgPool2d, HardwareElementwiseAdd, HardwareRelu, HardwareRelu6, InputQuantizer, OutputDequantizer
from .model_introspector import StdModelInspector

__all__ = [
    "HardwareConv2d",
    "HardwareLinear",
    "HardwareMaxPool2d",
    "HardwareGAP",
    "HardwareAdaptiveAvgPool2d",
    "HardwareElementwiseAdd",
    "HardwareRelu",
    "HardwareRelu6",
    "InputQuantizer",
    "OutputDequantizer",
    "StdModelInspector"
]
