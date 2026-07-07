import torch
import torchvision.models as models
from fixquant.graph.qat_processor import QatProcessor
import yaml

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

config = yaml.safe_load(open("configs/quant_config.yaml"))
config["is_qat"] = True

processor = QatProcessor(model, config)
qat_model = processor.quantize()
print("Successfully quantized MobileNetV2!")
