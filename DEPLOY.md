# Fixed-Point Quantization Model Deployment Tool

The script (`deploy_eval.py`) orchestrates the process of taking a pre-trained PyTorch model, applying Quantization-Aware Training (QAT) transformations, loading QAT-finetuned weights, and then extracting quantized model parameters in formats suitable for hardware deployment or further analysis. It also includes functionality to extract specific layer parameters (optionally subsetted) and can be extended to generate test vectors.

## Features

*   Supports standard torchvision models (e.g., ResNet50, ResNet18, VGG16) and custom CIFAR models.
*   Integrates a Quantization-Aware Training (QAT) pipeline using `QatProcessor`.
*   Loads pre-trained QAT weights onto the quantized model.
*   Converts the QAT model into a "standard" inference-ready format using `InferProcessor`.
*   Generates a `qconfig` dictionary detailing quantization parameters (bitwidth, fractional bits) for each layer.
*   Exports all model weights and biases to a single binary `.data` file in `int8` format.
*   Extracts parameters (weights and biases) from a *specific layer*, optionally taking a *subset* of these parameters, quantizes them, saves to a `.data` file, and returns the original floating-point subset.
*   Configurable via command-line arguments and a `quant_config.yaml` file.
*   Includes (commented out) example for image preprocessing, inference with an emulated model, and saving input/output tensors for hardware verification.

## Prerequisites

1.  **Python Environment:**
    *   Python 3.8+
    *   PyTorch (with CUDA support if using GPU)
    *   Torchvision
    *   Pillow (PIL)
    *   PyYAML
    *   NumPy
    A `requirements.txt` is available.

2.  **Custom Modules:** Ensure the following custom modules are in your `PYTHONPATH` or accessible from the script's directory:
    *   `models.cifar_models`
    *   `quantization.fix_ops`
    *   `quantization.utils.graph_trace`
    *   `quantization.utils.inference_mod`
    *   `data_providers.imagenet`
    *   `data_providers.cifar10`
    *   `run_manager`

3.  **Datasets (Optional, for data provider paths):**
    *   **ImageNet:** The script sets a default path. Update `ImagenetDataProvider.DEFAULT_PATH` if your location differs.
    *   **CIFAR-10:** (If using CIFAR models) Ensure the data provider can access it.

4.  **QAT Configuration:**
    *   A `quantization/utils/quant_config.yaml` file is required to define QAT settings.

5.  **Pre-trained QAT Weights:**
    *   The script expects QAT-finetuned weights, for example, at `qat_models/checkpoint/resnet50_best.pth.tar`. Update this path if necessary.

## Directory Structure
`````
├── deploy_eval.py # The deployment script
├── quantization/
│ ├── utils/
│ │ ├── quant_config.yaml # QAT configuration file
│ │ ├── graph_trace.py # Contains QatProcessor
│ │ └── inference_mod.py # Contains InferProcessor
│ └── fix_ops.py # Contains to_int_tensor, etc.
├── models/
│ └── cifar_models.py # Custom CIFAR models
├── data_providers/
│ ├── imagenet.py
│ └── cifar10.py
├── run_manager.py
├── qat_models/
│ └── checkpoint/
│ └── resnet50_best.pth.tar # Example QAT trained weights
├── hw_outputs/ # Directory for test image/output (if uncommented)
│ ├── test_image.data
│ └── ref_output.data
└── README.md
`````
## Configuration

1.  **`quantization/utils/quant_config.yaml`:**
    This YAML file defines the quantization strategy for different layer types, including bit-widths and 
whether to quantize weights, activations, and biases. Modify this file to change how QAT is applied.
2.  **Dataset Paths:**
    If using ImageNet, update the default path in `deploy.py` within the `if platform.system() == ...` block to point to your ImageNet dataset location.
3.  **QAT Checkpoint Path:**
    Update the path in `Qatprocessor.load_qat_weights('qat_models/checkpoint/resnet50_best.pth.tar')` 
    to point to your actual trained QAT model checkpoint.

## Usage

The script is run from the command line:

```bash
python deploy_eval.py [OPTIONS]
```

### Command-Line Options:

*   `--test_batch_size`: Batch size for testing/validation (default: 100).
*   `--valid_size`: Validation set size (default: None).
*   `--test_criterion`: Test criterion (default: "ce").
*   `--n_worker`: Number of data loading workers (default: 8).
*   `--pin-memory`: Use pinned memory for data loading (default: True).
*   `--device`: Device to use (e.g., "cuda", "cpu") (default: "cuda" if available).
*   `--gpus`: GPU IDs to use, comma-separated (e.g., "0,1") (default: "0").
*   `--dataset`: Dataset to use ("imagenet", "cifar10", "cifar100") (default: "imagenet").
*   `--dataroot`: Root directory for the dataset (default: path for ImageNet).
*   `--display_freq`: Frequency to display training metrics (default: 100).
*   `--validation_frequency`: Frequency to validate model (default: 1).
*   `--save_dir`: Directory to save trained QAT models (default: './qat_models').
*   `--output_dir`: Directory to save QAT results (default: 'qat_result').
*   `--manual_seed`: Manual seed for reproducibility (default: 0).

Most of these options are relevant if you uncomment the training/validation parts of the script using `RunManager`. For the current primary use (parameter extraction), `dataroot`, `dataset` (for model selection logic), and `gpus`/`device` are most relevant.

## Key Script Operations

1.  **Argument Parsing & Setup:** Parses command-line arguments and sets up device configurations.
2.  **Model Loading:**
    *   Loads a specified torchvision model (e.g., `resnet50`) with default pre-trained weights OR a custom CIFAR model.
3.  **QAT Processing (`QatProcessor`):**
    *   Initializes `QatProcessor` with the model and `quant_config.yaml`.
    *   `Qatprocessor.quantize()`: Modifies the model by replacing layers with their quantized equivalents (e.g., `QuantConv2d`, `QuantLinear`) and inserts fake quantization nodes.
    *   `Qatprocessor.freeze()`: "Freezes" batch norm statistics. **Important Note:** The script mentions a specific order for freezing and loading weights for different models (ResNet50 vs. VGG16/ResNet18). Pay attention to this if you change models.
    *   `Qatprocessor.load_qat_weights()`: Loads the weights from a QAT-finetuned checkpoint onto the quantized model. These weights are still floating-point but have been trained with fake quantization in the loop.
4.  **Inference Processor (`InferProcessor`):**
    *   Initializes `InferProcessor` with the QAT-processed model and `quant_config.yaml`.
    *   `infer_processor.convert_to_std_model()`: Converts the QAT model (with `QuantConv2d`, etc.) to a model with standard `nn.Conv2d`, `nn.Linear` layers but populates them with attributes like `frac_weight`, `frac_bias`, `frac_act` based on the QAT process. The weights and biases themselves are still floating-point at this stage but represent the "learned" quantized values.
    *   `infer_processor.generate_qconfig()`: Traverses the standard model graph to determine quantization parameters (bitwidth, fractional bits for inputs, weights, biases, outputs) for each layer and prints this configuration.
5.  **Parameter Export:**
    *   `infer_processor.export_weights_to_file()`: Iterates through all Conv2d and Linear layers in the `stdm` (standard model). For each layer:
        *   It retrieves the floating-point `weight` and `bias`.
        *   It uses `layer.frac_weight` and `layer.frac_bias` (and a fixed `n_bits=8`).
        *   It calls `to_int_tensor()` to convert these float parameters to `int8` fixed-point tensors.
        *   All these `int8` tensors are concatenated and saved to a binary file (typically `weights.data` or similar, name defined within `InferProcessor`).
    *   `infer_processor.extract_and_subset_layer_parameters()`:
        *   Extracts parameters from a *user-specified layer* (e.g., "conv1").
        *   Optionally takes a *subset* of the weights (e.g., first 16 output channels) based on `target_weight_shape`. The bias is subsetted accordingly.
        *   Returns the **original floating-point** subsetted weights and biases (for generating reference outputs).
        *   Quantizes the subsetted float parameters to `int8` using `to_int_tensor()`.
        *   Saves these quantized subsetted parameters to a specified `.data` file.
6.  **Test Image Processing & Inference (TODO):**
    *   Will do the following:
        *   Loading and preprocessing a test image (`new.JPEG`).
        *   Applying fake quantization or `to_int_tensor` to the input image to simulate quantized input.
        *   Saving the processed image to `hw_outputs/test_image.data`.
        *   Performing inference using `emu_model` (emulation model for emulation).
        *   Quantizing the model's output.
        *   Saving the quantized prediction to `hw_outputs/ref_output.data`.
    This part will generate test vectors for hardware verification.

7.  **Validation (Optional):**
    *   Using `RunManager` to perform model validation.

## Outputs

*   **Console Output:**
    *   The structure of the QAT-processed model.
    *   The generated `qconfig` dictionary.
    *   Log messages detailing the extraction process and returned floating-point tensor shapes from `extract_and_subset_layer_parameters`.
*   **Files Generated (by default, paths may vary based on internal implementation of `InferProcessor`):**
    *   A binary file containing all `int8` weights and biases of the model (e.g., `weights.data`), generated by `infer_processor.export_weights_to_file()`.
    *   A binary file for the subsetted layer, e.g., `conv1_subset.data`, generated by `infer_processor.extract_and_subset_layer_parameters()`.
    *   If the test image processing section is uncommented:
        *   `hw_outputs/test_image.data`: The processed `int8` input image.
        *   `hw_outputs/ref_output.data`: The `int8` reference output from the model.
