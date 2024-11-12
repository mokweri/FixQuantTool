import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from quantization.fix_ops import find_fix_pos


def calculate_fractional_length(std_dev, signed=True):
    """
    Calculate the optimal fractional length (FL) based on the standard deviation.
    The formula differs for signed and unsigned fixed-point numbers.

    Args:
    - std_dev (torch.Tensor): The standard deviation of the weights.
    - signed (bool): Whether the values are signed or unsigned. Default is True.

    Returns:
    - fractional_length (torch.Tensor): Calculated optimal fractional length (FL).
    """
    base = 40 if signed else 70
    # Calculating fractional length using the given formula: FL* ≈ log2(base / std_dev)
    fractional_length = torch.floor(torch.log2(torch.tensor(base) / std_dev))

    return fractional_length


def get_fractional_lengths(model, method, signed=True):
    weight_fractional_lengths = []
    bias_fractional_lengths = []
    layer_names = []

    # Go through each layer in the model
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Calculate FL for weights
            if method == "standard":
                std_dev_weights = layer.weight.std()
                fl_weights = calculate_fractional_length(std_dev_weights, signed=signed).item()
            elif method == "custom":
                fl_weights = find_fix_pos(layer.weight, bit_width=8, scope=1, method=2)  # Expecting integer output
            weight_fractional_lengths.append(fl_weights)

            # Check if layer has bias and calculate its FL
            if layer.bias is not None:
                if method == "standard":
                    std_dev_bias = layer.bias.std()
                    fl_bias = calculate_fractional_length(std_dev_bias, signed=signed).item()
                elif method == "custom":
                    fl_bias = find_fix_pos(layer.bias, bit_width=8, scope=2, method=2)  # Expecting integer output
                bias_fractional_lengths.append(fl_bias)
            else:
                bias_fractional_lengths.append(None)

            layer_names.append(name)

    return layer_names, weight_fractional_lengths, bias_fractional_lengths


def calculate_quantization_error(input_tensor, fl):
    # Simulate quantization by scaling according to FL, then calculating squared error
    # scale = 2 ** fl
    # quantized = torch.round(input_tensor * scale) / scale
    # error = torch.sum((input_tensor - quantized) ** 2).item()

    # scale = 2 ** fl
    # quantized = torch.round(input_tensor * scale) / scale

    scale = 2 ** fl
    wl = 8
    signed = True
    # Define clipping bounds for 8-bit representation
    if signed:
        min_val = -2 ** (wl - 1)
        max_val = 2 ** (wl - 1) - 1
    else:
        min_val = 0
        max_val = 2 ** wl - 1

    if signed:
        max_v = (1 << (wl - 1)) - 1
        min_v = -max_v - 1
    else:
        min_v = 0
        max_v = (1 << wl) - 1

        # Apply scaling, clipping, rounding, and de-scaling
    quantized = torch.clamp(torch.round(input_tensor * scale), min_v, max_v) / scale

    mse_error = torch.mean((input_tensor - quantized) ** 2).item()
    return mse_error


def get_quantization_errors(model, method, signed=True):
    weight_errors = []
    bias_errors = []
    layer_names = []

    # Go through each layer in the model
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Calculate FL for weights using specified method
            if method == "standard":
                std_dev_weights = layer.weight.std()
                fl_weights = calculate_fractional_length(std_dev_weights, signed=signed)
            elif method == "custom":
                fl_weights = find_fix_pos(layer.weight, bit_width=8, scope=1, method=2)

            # Calculate quantization error for weights
            weight_error = calculate_quantization_error(layer.weight, fl_weights)
            weight_errors.append(weight_error)

            # Check if layer has bias and calculate quantization error
            if layer.bias is not None:
                if method == "standard":
                    std_dev_bias = layer.bias.std()
                    fl_bias = calculate_fractional_length(std_dev_bias, signed=signed)
                elif method == "custom":
                    fl_bias = find_fix_pos(layer.bias, bit_width=8, scope=2, method=2)

                # Calculate quantization error for biases
                bias_error = calculate_quantization_error(layer.bias, fl_bias)
                bias_errors.append(bias_error)
            else:
                bias_errors.append(None)

            layer_names.append(name)

    return layer_names, weight_errors, bias_errors


if __name__ == '__main__':

    model = models.resnet50(pretrained=True)

    # layer_names, weight_fl_standard, bias_fl_standard = get_fractional_lengths(model, method="standard", signed=True)
    # _, weight_fl_custom, bias_fl_custom = get_fractional_lengths(model, method="custom", signed=True)

    # Plotting both methods for comparison
    # fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    #
    # # Plot for standard FL calculation
    # axes[0].bar(range(len(weight_fl_standard)), weight_fl_standard, label="Weights", alpha=0.6)
    # bias_positions_standard = [i for i, v in enumerate(bias_fl_standard) if v is not None]
    # bias_values_standard = [v for v in bias_fl_standard if v is not None]
    # axes[0].bar(bias_positions_standard, bias_values_standard, label="Biases", alpha=0.6)
    # axes[0].set_xticks(range(len(layer_names)))
    # axes[0].set_xticklabels(layer_names, rotation=90)
    # axes[0].set_xlabel("Layers")
    # axes[0].set_ylabel("Fractional Length (FL)")
    # axes[0].set_title("Fractional Length (FL) for Weights and Biases (Standard Method)")
    # axes[0].legend()
    #
    # # Plot for custom FL calculation
    # axes[1].bar(range(len(weight_fl_custom)), weight_fl_custom, label="Weights", alpha=0.6)
    # bias_positions_custom = [i for i, v in enumerate(bias_fl_custom) if v is not None]
    # bias_values_custom = [v for v in bias_fl_custom if v is not None]
    # axes[1].bar(bias_positions_custom, bias_values_custom, label="Biases", alpha=0.6)
    # axes[1].set_xticks(range(len(layer_names)))
    # axes[1].set_xticklabels(layer_names, rotation=90)
    # axes[1].set_xlabel("Layers")
    # axes[1].set_ylabel("Fractional Length (FL)")
    # axes[1].set_title("Fractional Length (FL) for Weights and Biases (Custom Method)")
    # axes[1].legend()
    #
    # plt.tight_layout()
    # plt.show()

    # Calculate quantization errors for each layer using both methods
    layer_names, weight_errors_standard, bias_errors_standard = get_quantization_errors(model, method="standard",
                                                                                        signed=True)
    _, weight_errors_custom, bias_errors_custom = get_quantization_errors(model, method="custom", signed=True)

    # print(weight_errors_standard)
    # print(weight_errors_custom)

    # Combine all errors to find the global maximum for normalization
    all_errors = weight_errors_standard + list(filter(None, bias_errors_standard)) + \
                 weight_errors_custom + list(filter(None, bias_errors_custom))
    global_max_error = max(all_errors) if all_errors else 1

    # Normalize all errors using the global maximum
    weight_errors_standard = [e / global_max_error for e in weight_errors_standard]
    bias_errors_standard = [e / global_max_error if e is not None else None for e in bias_errors_standard]
    weight_errors_custom = [e / global_max_error for e in weight_errors_custom]
    bias_errors_custom = [e / global_max_error if e is not None else None for e in bias_errors_custom]

    # Plotting quantization errors for comparison
    # Plotting normalized quantization errors for comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # Set a common y-axis limit for both plots
    y_max = 1.1  # Since errors are normalized, the max should be 1

    # Plot for standard quantization error
    axes[0].bar(range(len(weight_errors_standard)), weight_errors_standard, label="Weights", alpha=0.6)
    bias_positions_standard = [i for i, v in enumerate(bias_errors_standard) if v is not None]
    bias_values_standard = [v for v in bias_errors_standard if v is not None]
    axes[0].bar(bias_positions_standard, bias_values_standard, label="Biases", alpha=0.6)
    axes[0].set_ylim(0, y_max)
    axes[0].set_xticks(range(len(layer_names)))
    axes[0].set_xticklabels(layer_names, rotation=90)
    axes[0].set_xlabel("Layers")
    axes[0].set_ylabel("Normalized Quantization Error")
    axes[0].set_title("Normalized Quantization Error for Weights and Biases (Standard Method)")
    axes[0].legend()

    # Plot for custom quantization error
    axes[1].bar(range(len(weight_errors_custom)), weight_errors_custom, label="Weights", alpha=0.6)
    bias_positions_custom = [i for i, v in enumerate(bias_errors_custom) if v is not None]
    bias_values_custom = [v for v in bias_errors_custom if v is not None]
    axes[1].bar(bias_positions_custom, bias_values_custom, label="Biases", alpha=0.6)
    axes[1].set_ylim(0, y_max)
    axes[1].set_xticks(range(len(layer_names)))
    axes[1].set_xticklabels(layer_names, rotation=90)
    axes[1].set_xlabel("Layers")
    axes[1].set_ylabel("Normalized Quantization Error")
    axes[1].set_title("Normalized Quantization Error for Weights and Biases (Custom Method)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()