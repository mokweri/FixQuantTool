import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from quantization.fix_ops import fix_quantize_tensor


class FakeQuantizer(nn.Module):
    """Simulate the quantize and dequantize operations in training time.

      In general, the output of this module is given by
      x_out = (clamp(round(x / scale + zero_point), quant_min, quant_max) - zero_point) * scale
      See https://arxiv.org/pdf/1903.08066.pdf

      We use symmetric quantization and power-of-2 scaling (Fixed point quantization). That is,
        zero_point = 0,
        quant_min = -2^(bitwidth - 1),
        quant_max = 2^(bitwidth - 1) - 1
    """
    _version = 2

    def __init__(self, bitwidth):
        super(FakeQuantizer, self).__init__()
        # quant_enabled is registered as buffer to support their replication in DDP.
        # Data type is uint8 because NCCL does not support bool tensors.
        self.register_buffer('quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('bitwidth', torch.tensor([bitwidth], dtype=torch.uint8))
        self.register_buffer('domain', torch.tensor([2 ** (bitwidth - 1)]).float())

    def forward(self, x):
        raise NotImplementedError(
            'Do not use FakeQuantizer directly, please use its derivatives.')

    # PyTorch has been using _save_to_state_dict since 1.2.0.
    # See https://github.com/pytorch/pytorch/blob/v1.2.0/torch/nn/modules/module.py.
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(FakeQuantizer, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination.pop(prefix + 'quant_enabled')
        destination.pop(prefix + 'domain')
        destination.pop(prefix + 'bitwidth')


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Remove 'bitwidth', 'quant_enabled', and 'domain' from state_dict if present
        ignored_params = ['bitwidth', 'quant_enabled', 'domain']
        ignored_keys = {prefix + name for name in ignored_params}

        missing_keys[:] = [key for key in missing_keys if key not in ignored_keys]
        unexpected_keys[:] = [key for key in unexpected_keys if key not in ignored_keys]
        # Temporarily override strict mode if missing keys should be ignored
        if strict:
            strict = False
        # # Call the parent class's _load_from_state_dict
        super(FakeQuantizer, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                         missing_keys, unexpected_keys, error_msgs)
        # Check for any unexpected keys and print warnings
        for key in unexpected_keys:
            if key in ignored_keys:
                print('[WARNING] Unexpected key in state dict:', key)


class TQTQuantizer(FakeQuantizer):
    def __init__(self, bitwidth, tensor_type, method=None):
        super(TQTQuantizer, self).__init__(bitwidth)

        valid_tensor_types = ['weight', 'act']
        if tensor_type not in valid_tensor_types:
            raise ValueError(
                "'tensor_type' must be one of {}".format(valid_tensor_types))
        self.tensor_type = tensor_type
        if method is not None:
            self.method = method
        else:
            self.method = 3 if tensor_type == 'weight' else 2

        self.quantize_fn_cls = TQTQuantize
        self.log_threshold = nn.Parameter(torch.tensor([0.0]))
        self.register_buffer('warmup_enabled', torch.tensor([1], dtype=torch.uint8))

        self._forward_fn = self._quantize_with_warmup

    def _init_threshold(self, x):
        """See Table 2 in https://arxiv.org/pdf/1903.08066.pdf"""

        def _max(x):
            return np.max(np.abs(x))

        def _3sd(x):
            y = x.astype(np.float32) if x.dtype == np.float16 else x
            return np.abs(np.mean(y + 1e-6)) + 3 * np.std(y)

        def _kl_j(x):
            """
            Ref paper (Algorithm 1):
            "Quantizing Convolutional Neural Networks for Low-Power
            High-Throughput Inference Engines" - Sean Settle et al.
            https://arxiv.org/pdf/1805.07941.pdf
            """

            def calculate_kl_j(x, y):
                return np.sum((x - y) * np.log2(x / y))

            mn = 0
            mx = np.max(np.abs(x))
            y = x.astype(np.float32) if x.dtype == np.float16 else x
            hist, bin_edges = np.histogram((np.abs(y)),
                                           'sqrt',
                                           range=(mn, mx),
                                           density=True)
            hist = hist.astype(x.dtype)
            bin_edges = bin_edges.astype(x.dtype)
            pdf = hist / np.sum(hist)
            cdf = np.cumsum(pdf)
            n = pow(2, self.bitwidth.item() - 1)
            threshold = []
            d = []
            if n + 1 > len(bin_edges) - 1:
                return bin_edges[(-1)]
            else:
                for i in range(n + 1, len(bin_edges), 1):
                    threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
                    threshold = np.concatenate((threshold, [threshold_tmp]))
                    p = np.copy(cdf)
                    p[i - 1:] = 1
                    x = np.linspace(0.0, 1.0, n)
                    xp = np.linspace(0.0, 1.0, i)
                    fp = p[:i]
                    p_interp = np.interp(x, xp, fp)
                    x = np.linspace(0.0, 1.0, i)
                    xp = np.linspace(0.0, 1.0, n)
                    fp = p_interp
                    q_interp = np.interp(x, xp, fp)
                    q = np.copy(p)
                    q[:i] = q_interp
                    d_tmp = calculate_kl_j(cdf[np.nonzero(cdf)], q[np.nonzero(cdf)])
                    d = np.concatenate((d, [d_tmp]))

                return threshold[np.argmin(d)]

        init_scheme = {'weight': _3sd, 'act': _kl_j}
        # init_scheme = {'weight': _max, 'act': _kl_j}
        data = x.detach().cpu().numpy()
        th = init_scheme[self.tensor_type](data)

        return torch.tensor([th], dtype=x.dtype, device=x.device)

    def _forward_pass_input(self, x, log_threshold, domain, method):
        print("Just to let you know, the quantizer is pybassed here")
        return x

    def _quantize(self, x, log_threshold, domain, method):
        return self.quantize_fn_cls.apply(x, log_threshold, domain, method)

    def _quantize_with_warmup(self, x, log_threshold, domain, method):
        self.disable_warmup()
        log_threshold.data[0] = torch.log2(self._init_threshold(x))[0]
        return self._quantize(x, log_threshold, domain, method)

    def forward(self, x):
        return self._forward_fn(x, self.log_threshold, self.domain, self.method)

    def enable_warmup(self, enabled=True):
        self.warmup_enabled[0] = 1 if enabled else 0
        self._forward_fn = self._quantize_with_warmup if enabled else self._quantize
        return self

    def disable_warmup(self):
        return self.enable_warmup(False)

    def freeze_quant(self, frozen=True):
        self.log_threshold.requires_grad = (not frozen)

    def unfreeze_quant(self):
        self.freeze_quant(False)

    def extra_repr(self):
        return 'quant_enabled={}, bitwidth={}, method={}'.format(
            self.quant_enabled, self.bitwidth, self.method)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(TQTQuantizer, self)._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(TQTQuantizer, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys, unexpected_keys, error_msgs)
        self._forward_fn = self._quantize_with_warmup if self.warmup_enabled[0] == 1 else self._quantize
        if self.quant_enabled[0] == 0:
            self._forward_fn = self._forward_pass_input

    def export_quant_info(self):
        """Export trained threshold to quant info [bitwidth, fp].
            (1) TQT: qx = clip(round(fx / scale)) * scale, scale = 2^ceil(log2t) / 2^(b-1)
            (2) fixpoint op: qx = clip(round(fx * scale)) * (1 / scale), scale = 2^fp
            Let (1) equals (2), we can get
            (3): 2^(b-1) / 2^ceil(log2t) = 2^fp
             => fp = b - 1 - ceil(log2t)
        """
        bitwidth = self.bitwidth.item()
        ceil_log2t = torch.ceil(self.log_threshold).item()
        return [bitwidth, int(bitwidth - 1 - ceil_log2t)]


class TQTQuantize(torch.autograd.Function):
  """Trained Quantization Thresholds.
     See https://arxiv.org/pdf/1903.08066.pdf
  """

  @staticmethod
  def forward(ctx, x, logt, domain, method):
    #scale = torch.pow(torch.tensor(2.0, device=x.device), torch.ceil(logt)) / domain
    #scale = torch.pow(2.0, torch.ceil(logt)) / domain
    scale = 2**(torch.ceil(logt)) / domain
    quant_max = domain - 1
    quant_min = -domain

    ctx.save_for_backward(x, scale, quant_max, quant_min, logt)

    x = x.clone()
    return fix_quantize_tensor(x, quant_min, quant_max, scale, 0, 2)


  @staticmethod
  def backward(ctx, grad_output):
    x, scale, quant_max, quant_min, logt = ctx.saved_tensors

    scaled_x = x / scale

    # Python equivalent to NndctFixNeuron rounding implementation which is consistent with hardware runtime.
    # Round -1.5 to -1 instead of -2.
    rounded_scaled_x = torch.where(
        (scaled_x < 0) & (scaled_x - torch.floor(scaled_x) == 0.5),
        torch.ceil(scaled_x), torch.round(scaled_x))

    is_lt_min = rounded_scaled_x < quant_min
    is_gt_max = rounded_scaled_x > quant_max
    is_ge_min_and_le_max = ~is_lt_min & ~is_gt_max

    # Equation (7) in section 3.3
    #grad_logt = torch.ones(grad_output.shape, dtype=grad_output.dtype, device=grad_output.device) * scale * math.log(2)
    grad_logt = grad_output * scale * math.log(2)
    grad_logt = torch.where(is_ge_min_and_le_max,
                            grad_logt * (rounded_scaled_x - scaled_x),
                            grad_logt)
    grad_logt = torch.where(is_lt_min, grad_logt * quant_min, grad_logt)
    grad_logt = torch.where(is_gt_max, grad_logt * quant_max, grad_logt)
    grad_logt = grad_logt.sum().expand_as(logt)

    # Equation (8)
    grad_x = grad_output.clone()
    grad_x = torch.where(is_ge_min_and_le_max, grad_x, 0 * grad_x)

    return grad_x, grad_logt, None, None


if __name__ == '__main__':
    tqtq = TQTQuantizer(bitwidth=8, tensor_type='weight')
    float_tensor = torch.tensor([[0.5, -0.75, 1.25], [0.1, 0.3, -0.2]], dtype=torch.float32)
    print(tqtq.state_dict())
    qtensor = tqtq.forward(float_tensor)

    print(qtensor)
    print(tqtq.export_quant_info())
    torch.save(tqtq.state_dict(), 'tqt.pth')
    tqtq.load_state_dict(torch.load('tqt.pth'))

    # print(tqtq.state_dict())



