import torch
import numpy as np
import logging
from typing import Optional, Tuple
from quantization.fix_ops import to_int_tensor


class ModelParameterExtractor:
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger if logger else logging.getLogger(__name__)

    def extract_and_subset_layer_parameters(
            self,
            layer_name: str,
            output_filename_weights: str,
            output_filename_biases: str,
            target_weight_shape: Optional[Tuple[int, ...]] = None,
            n_bits_out: int = 8
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, int]:
        """
        Extracts weights and biases from a specific layer, optionally subsets them,
        quantizes them, saves to separate files, and returns the original floating-point tensors.

        Returns:
            tuple: (fp_weights, fp_biases, frac_w, frac_b)
        """
        self.logger.info(f"Extracting parameters from layer '{layer_name}'")
        if target_weight_shape:
            self.logger.info(f"Target subset weight shape: {target_weight_shape}")

        module_to_extract = dict(self.model.named_modules()).get(layer_name)

        if module_to_extract is None:
            self.logger.error(f"Layer '{layer_name}' not found.")
            raise ValueError(f"Layer '{layer_name}' not found.")
        if not isinstance(module_to_extract, (torch.nn.Conv2d, torch.nn.Linear)):
            self.logger.error(f"Layer '{layer_name}' is {type(module_to_extract).__name__}, not Conv2d/Linear.")
            raise TypeError(f"Layer '{layer_name}' is not a Conv2d or Linear layer.")

        return_fp_weight: Optional[torch.Tensor] = None
        return_fp_bias: Optional[torch.Tensor] = None
        frac_w = 0
        frac_b = 0

        # Process Weights
        if hasattr(module_to_extract, 'weight') and module_to_extract.weight is not None:
            original_fp_weights = module_to_extract.weight.data.detach().clone()
            self.logger.debug(f"Original FP weight shape: {original_fp_weights.shape}")

            # Subset weights if target shape is provided
            return_fp_weight = self._subset_tensor(original_fp_weights, target_weight_shape)

            # Get fractional bits for weights
            frac_w = int(getattr(module_to_extract, 'frac_weight', 0))
            if not hasattr(module_to_extract, 'frac_weight'):
                self.logger.warning(f"Layer '{layer_name}' weights missing 'frac_weight', using default 0.")

            # Quantize weights
            quantized_weights = to_int_tensor(
                return_fp_weight, signed=True, n_bits=n_bits_out, n_frac=frac_w
            )

            if quantized_weights is not None:
                self._save_to_file(quantized_weights.cpu().numpy().astype('int8'), output_filename_weights)
                self.logger.info(f"Saved quantized weights to '{output_filename_weights}'")
            else:
                self.logger.error(f"Quantization of weights for '{layer_name}' resulted in None.")
        else:
            self.logger.warning(f"Layer '{layer_name}' has no 'weight' or it's None.")

        # Process Biases
        if hasattr(module_to_extract, 'bias') and module_to_extract.bias is not None:
            original_fp_bias = module_to_extract.bias.data.detach().clone()
            self.logger.debug(f"Original FP bias shape: {original_fp_bias.shape}")

            # Subset bias if weights were subset (match first dimension)
            if return_fp_weight is not None and target_weight_shape is not None:
                target_bias_len = return_fp_weight.shape[0]
                if target_bias_len < original_fp_bias.shape[0]:
                    return_fp_bias = original_fp_bias[:target_bias_len]
                    self.logger.info(f"Subsetted FP bias to length: {return_fp_bias.shape[0]}")
                else:
                    return_fp_bias = original_fp_bias
            else:
                return_fp_bias = original_fp_bias

            # Get fractional bits for bias
            frac_b = int(getattr(module_to_extract, 'frac_bias', 0))
            if not hasattr(module_to_extract, 'frac_bias'):
                self.logger.warning(f"Layer '{layer_name}' bias missing 'frac_bias', using default 0.")

            # Quantize bias
            quantized_bias = to_int_tensor(
                return_fp_bias, signed=True, n_bits=n_bits_out, n_frac=frac_b
            )

            if quantized_bias is not None:
                self._save_to_file(quantized_bias.cpu().numpy().astype('int8'), output_filename_biases)
                self.logger.info(f"Saved quantized biases to '{output_filename_biases}'")
            else:
                self.logger.error(f"Quantization of bias for '{layer_name}' resulted in None.")
        else:
            self.logger.debug(f"Layer '{layer_name}' has no 'bias' or it's None.")
            # Create empty bias file
            self._save_to_file(None, output_filename_biases)

        return frac_w, frac_b

    def _subset_tensor(self, tensor: torch.Tensor, target_shape: Optional[Tuple[int, ...]]) -> torch.Tensor:
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
        self.logger.info(f"Subsetted tensor from {tensor.shape} to {subset_tensor.shape}")
        return subset_tensor

    def _save_to_file(self, data: Optional[np.ndarray], filename: str):
        """Helper function to save data to file."""
        try:
            if data is not None:
                data.tofile(filename)
                self.logger.info(f"Successfully wrote {data.size} elements to '{filename}'")
            else:
                # Create empty file
                with open(filename, 'wb') as f:
                    pass
                self.logger.info(f"Created empty file '{filename}'")
        except IOError as e:
            self.logger.error(f"Failed to write to file '{filename}': {e}")
            raise