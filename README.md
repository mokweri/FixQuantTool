# FixQuantTool: An End-to-End Fixed-Point Quantization Workflow for FPGAs

## Introduction

FixQuantTool is a comprehensive framework designed to bridge the gap between software-based deep learning model 
quantization and efficient deployment on Field-Programmable Gate Arrays (FPGAs). 
Deploying deep learning models on resource-constrained hardware like FPGAs presents challenges, 
as standard quantization techniques often do not map optimally to the target hardware. 
This tool implements an end-to-end workflow for fixed-point quantization, integrating hardware emulation 
directly into the Quantization-Aware Training (QAT) process to ensure that quantized models are optimized 
for real-world FPGA deployment.

This project is based on the research presented in the paper:
**"Bridging Quantization and Deployment: A Fixed-Point Workflow for FPGA Accelerators"**
*Obed M. Mogaka, Håkan Forsberg, Masoud Daneshtalab*
DOI: `10.1109/DDECS63720.2025.11006791`

Our methodology aims to resolve discrepancies between software quantization and hardware behavior, 
providing a practical solution for edge device applications.

## Key Features

* **Efficient Fixed-Point Quantization:** Utilizes binary-point scaling, optimally mapping model weights and activations to integer values for efficient bit-shifting operations on FPGAs.
* **Quantization-Aware Training (QAT):** Implements QAT to fine-tune models, converting weights and activations to fixed-point representation while learning optimal fractional lengths. It uses a TQT-like approach for learning quantization thresholds.
* **Hardware Emulation in QAT:** Integrates a hardware emulation engine that considers various HLS rounding modes (RND, RND\_ZERO, RND\_MIN\_INF, RND\_CONV, TRN). This helps identify optimal layer-specific configurations and minimize quantization errors when mapping to hardware.
* **Batch Normalization (BN) Folding:** Implements BN folding by combining BN layers with preceding convolutional layers to reduce inference overhead. A two-forward-pass strategy is used during training to manage statistics before freezing the BN layers.
* **Graph Editing and Optimization:** Leverages `torch.fx` for tracing and transforming model graphs, replacing standard modules with their quantized counterparts to create a leaner graph for quantization.
* **Flexible Deployment Interface:** Provides an interface to generate hardware-specific configuration files (quantization parameters, weights). Validated with the PipeCNN open-source FPGA accelerator and designed for extensibility to other FPGA platforms.
* **Model & Dataset Support:** Validated with ResNet and VGG models on CIFAR-10 and ImageNet datasets.

## Workflow Overview

The proposed end-to-end workflow generally consists of the following steps:

1.  **Input Model:** Start with a pre-trained PyTorch model.
2.  **Graph Editing and Optimization:** 
    * The model's computational graph is traced using `torch.fx`.
    * Batch Normalization layers are fused with their preceding convolutional layers.
    * Standard modules (convolution, pooling, linear) are replaced with their quantizable equivalents.
3.  **Quantization-Aware Training (QAT):**
    * The model undergoes fine-tuning to learn quantization parameters (fractional lengths via TQT approach).
    * Hardware emulation is performed during QAT to account for hardware-specific behaviors like rounding modes.
4.  **Deployment Output:**
    * The process generates quantized model weights and hardware configuration files suitable for FPGA deployment.

## Getting Started

1.  **Prerequisites:** Ensure you have Python and PyTorch installed. Install necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configuration:**
    * Quantization settings are primarily managed via `quantization/utils/quant_config.yaml`.
    * Dataset paths can be configured within the scripts (e.g., `qat_train.py`, `deploy_eval.py`) or data provider files.
3.  **Main Scripts:**
    * `qat_train.py`: Use this script for performing Quantization-Aware Training. See the script's internal argument parser or the original README for details on its arguments.
    * `deploy_eval.py`: Use this script for post-QAT processing, model conversion, parameter extraction, and evaluation. See `DEPLOY.MD` for more details.

## Directory Structure

A brief overview of the key directories:

* `./`: Contains main scripts like `train.py`, `qat_train.py`, `deploy_eval.py`.
* `data_providers/`: Modules for loading datasets (CIFAR, ImageNet).
* `docs/`: Contains detailed documentation for different aspects of the tool.
* `hw_fxp/`: Hardware-specific fixed-point emulation and testing utilities.
* `hw_outputs/`: Example output files for hardware, and a README explaining their format.
* `models/`: PyTorch model definitions, including custom CIFAR models.
* `qconfig_files/`: Example JSON configuration files for quantization parameters.
* `quantization/`: Core logic for quantization, including QAT modules, fixed-point operations, TQT quantizers, and graph transformation utilities.
* `run_manager/`: Manages training and evaluation runs.
* `scripts/`: Shell scripts for job submission (e.g., on SLURM).
* `utils/`: General utility functions.

## Documentation

For more detailed information, please refer to the following documents:

* **This README (`README.md`):** Provides a general overview of the FixQuantTool project.
* **[QAT Training Script Details (`QAT.md`)](QAT.md)**: Detailed instructions for using the `qat_train.py` script.
* **[Deployment and Evaluation (`DEPLOY.md`)](./DEPLOY.md):** Guide for using `deploy_eval.py` for model deployment preparation and evaluation.
* **[Fused Convolution-BN (`docs/conv_fused.md`)](./docs/conv_fused.md):** Information on the fused convolution and batch normalization modules.
* **[Quantized Modules (`docs/qmodules.md`)](/docs/qmodules.md):** Details about the custom quantized PyTorch modules.
* **[Trained Quantization Thresholds (`docs/tqt.md`)](./docs/tqt.md):** Explanation of the TQT-based quantizers.
* **[Hardware Output Files (`hw_outputs/Readme.md`)](./hw_outputs/Readme.md):** Description of the format and interpretation of binary files generated for hardware.

## Citation
If you use FixQuantTool or find the associated research paper helpful, please cite:
```bibtex
@INPROCEEDINGS{Mogaka2025DDECS,
  author={Mogaka, Obed M. and Forsberg, Håkan and Daneshtalab, Masoud},
  booktitle={2025 IEEE 28th International Symposium on Design and Diagnostics of Electronic Circuits and Systems (DDECS)}, 
  title={Bridging Quantization and Deployment: A Fixed-Point Workflow for FPGA Accelerators}, 
  year={2025},
  pages={to appear},
  doi={10.1109/DDECS63720.2025.11006791}
}
```

## Acknowledgments
This work was supported in part by the Swedish Innovation Agency VINNOVA projects FASTER-AI and AutoDeep, 
the European Union and Estonian Research Council via project TEM-TA138. 
The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), 
partially funded by the Swedish Research Council through grant agreement no. 2022-06725.