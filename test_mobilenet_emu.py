import torch
import torchvision.models as models
from fixquant.graph.qat_processor import QatProcessor
from fixquant.graph.inference_processor import InferProcessor
import yaml

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

config = yaml.safe_load(open("configs/quant_config.yaml"))
config["is_qat"] = True

processor = QatProcessor(model, config)
qat_model = processor.quantize()

infer_proc = InferProcessor(qat_model, config)
emu_model = infer_proc.convert_to_hardware_model()
print("Successfully converted MobileNetV2 to Emulation model!")
