# python
# File: `tools/archive/gen_resnet18_fc_testdata.py`
from pathlib import Path
import torchvision.models as tv_models
import hw_layer_test_gen as gen  # run from tools/archive/

# Monkeypatch: make calls to resnet50() produce a resnet18 instance
tv_models.resnet50 = lambda weights=None: tv_models.resnet18(weights=None)

# Target only the final fully-connected layer
gen.LAYER_SPECS = {
    "fc": "final fully-connected classifier (scores for 1000 classes)"
}

# Optional: point to a resnet18 QAT checkpoint if you have one in the repo
gen.MODEL_CHECKPOINT = str(gen.PROJECT_ROOT / "qat_models/checkpoint/resnet18_best.pth.tar")

# Ensure output base (defaults to `outputs/hw_data_files` already)
gen.BASE_OUT_DIR = str(gen.PROJECT_ROOT / "outputs/hw_data_files")

if __name__ == "__main__":
    gen.main()
