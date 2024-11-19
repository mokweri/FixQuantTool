import torch
import numpy as np
import math
import torch.nn.functional as F


def _init_threshold(x):
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
        n = pow(2, 8 - 1)
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
    return _kl_j(x)


def _init_threshold2(x):
    def _max(x):
        return torch.max(torch.abs(x))

    def _3sd(x):
        eps = 1e-8
        # x += eps
        mean_value = torch.tensor([x.mean().abs().data]) + eps
        std_value = x.std().data
        return mean_value + 3 * std_value

    def _kl_j(x, bitwidth=8):
        mn = 0
        mx = torch.max(torch.abs(x))
        x = x.to(torch.float32)  # Ensure float32 for precision
        x = x.to("cuda")

        def calculate_kl_divergence(p, q):
            mask = (p != 0) & (q != 0)
            return torch.sum(p[mask] * torch.log2(p[mask] / q[mask]))

        # Manually calculate histogram
        bins = int(np.sqrt(x.numel()))
        bin_edges = torch.linspace(mn, mx, bins + 1)
        hist = torch.histc(x.abs(), bins=bins, min=mn, max=mx)
        pdf = hist / hist.sum()
        cdf = torch.cumsum(pdf, dim=0)

        # Threshold and KL divergence calculations
        n = 2 ** (bitwidth - 1)
        threshold = []
        d = []

        if n + 1 > len(bin_edges) - 1:
            return bin_edges[-1]
        else:
            for i in range(n + 1, len(bin_edges)):
                threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
                threshold.append(threshold_tmp)

                # Copy and interpolate distributions
                p = cdf.clone()
                p[i:] = 1
                interp_x = torch.linspace(torch.tensor(0.0, device=x.device),
                                          torch.tensor(1.0, device=x.device), n, device=x.device)
                interp_fp = p[:i]
                xp = torch.linspace(torch.tensor(0.0, device=x.device),
                                    torch.tensor(1.0, device=x.device), i, device=x.device)
                p_interp = interp(interp_x, xp, interp_fp)

                q_interp = torch.zeros_like(p)
                q_interp[:i] = p_interp
                q_interp[i:] = p[i:]

                d_tmp = calculate_kl_divergence(cdf, q_interp)
                d.append(d_tmp.item())

            threshold_idx = torch.tensor(d).argmin()
            return threshold[threshold_idx]

    return _kl_j(x)

def _cdf_measure(x, y, measure_name='Kullback-Leibler-J'):
  """
    Ref paper:
    "Non-parametric Information-Theoretic Measures of One-Dimensional
    Distribution Functions from Continuous Time Series" - Paolo D Alberto et al.
    https://epubs.siam.org/doi/abs/10.1137/1.9781611972795.59
    https://epubs.siam.org/doi/pdf/10.1137/1.9781611972795.59

    measure_names_symm = ['Camberra', 'Chi-Squared', 'Cramer-von Mises', 'Euclidean',
               'Hellinger', 'Jin-L', 'Jensen-Shannon', 'Kolmogorov-Smirnov',
               'Kullback-Leibler-J', 'Variational']
    measure_names_asym = ['Jin-K', 'Kullback-Leibler-I']
    measure_names_excl = ['Bhattacharyya', 'Phi', 'Xi']
    """
  if measure_name == 'Bhattacharyya':
    return np.sum(np.sqrt(x * y))
  else:
    if measure_name == 'Camberra':
      return np.sum(np.abs(x - y) / (x + y))
    else:
      if measure_name == 'Chi-Squared':
        return np.sum(np.power(x - y, 2.0) / x)
      else:
        if measure_name == 'Cramer-von Mises':
          return np.sum(np.power(x - y, 2.0))
        else:
          if measure_name == 'Euclidean':
            return np.power(np.sum(np.power(x - y, 2.0)), 0.5)
          else:
            if measure_name == 'Hellinger':
              return np.power(np.sum(np.sqrt(x) - np.sqrt(y)), 2.0) / 2.0
            else:
              if measure_name == 'Jin-K':
                return _cdf_measure(x, (x + y) / 2.0, 'Kullback-Leibler-I')
              else:
                if measure_name == 'Jin-L':
                  return _cdf_measure(
                      x, (x + y) / 2.0, 'Kullback-Leibler-I') + _cdf_measure(
                          y, (x + y) / 2.0, 'Kullback-Leibler-I')
                if measure_name == 'Jensen-Shannon':
                  return (
                      _cdf_measure(x, (x + y) / 2.0, 'Kullback-Leibler-I') +
                      _cdf_measure(y,
                                   (x + y) / 2.0, 'Kullback-Leibler-I')) / 2.0
                if measure_name == 'Kolmogorov-Smirnov':
                  return np.max(np.abs(x - y))
              if measure_name == 'Kullback-Leibler-I':
                return np.sum(x * np.log2(x / y))
            if measure_name == 'Kullback-Leibler-J':
              return np.sum((x - y) * np.log2(x / y))
          if measure_name == 'Phi':
            return np.max(
                np.abs(x - y) /
                np.sqrt(np.minimum((x + y) / 2.0, 1 - (x + y) / 2.0)))
        if measure_name == 'Variational':
          return np.sum(np.abs(x - y))
      if measure_name == 'Xi':
        return np.max(
            np.abs(x - y) / np.sqrt((x + y) / 2.0 * (1 - (x + y) / 2.0)))
    return _cdf_measure(x, y, 'Kullback-Leibler-J')


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int = -1,
           extrapolate: str = 'linear') -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points.

    Implementation according to https://github.com/pytorch/pytorch/issues/50334

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

    Returns:
        The interpolated values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)

    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    indices = torch.searchsorted(xp, x, right=False)

    if extrapolate == 'constant':
        # Pad m and b to get constant values outside of xp range
        m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else:  # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)

    return values.movedim(-1, dim)


if __name__ == '__main__':

    # Example usage for testing
    class Example:
        def __init__(self, bitwidth=8, tensor_type='act'):
            self.bitwidth = torch.tensor([bitwidth])
            self.tensor_type = tensor_type

        def init_threshold(self, x):
            th = _init_threshold(x)
            return th

    class Example2:
        def __init__(self, bitwidth=8, tensor_type='act'):
            self.bitwidth = torch.tensor([bitwidth])
            self.tensor_type = tensor_type

        def init_threshold(self, x):
            return _init_threshold2(x)

    # Test setup
    weight_shape = (16, 3, 3, 3)
    # Generate the random weight tensor with a normal distribution
    x = torch.randn(weight_shape)
    x2 = x.clone()
    y = x2.numpy()

    example = Example(bitwidth=8, tensor_type='act')
    example2 = Example2(bitwidth=8, tensor_type='act')

    # Calculate threshold in PyTorch
    threshold_numpy = example.init_threshold(y)
    print("Threshold (Numpy):", threshold_numpy)

    threshold_torch = example2.init_threshold(x)
    print("Threshold (Torch):", threshold_torch)



    # xp = torch.linspace(0, 2*math.pi, 10)
    # fp = torch.sin(xp)
    # x = torch.linspace(0, 2*math.pi, 50)
    # interpolated_vals = interp(x, xp, fp, extrapolate='linear')
    # print(interpolated_vals)
    #
    # x = np.linspace(0, 2 * np.pi, 10)
    # y = np.sin(x)
    # xvals = np.linspace(0, 2 * np.pi, 50)
    # yinterp = np.interp(xvals, x, y)
    # intrp = np.interp(xvals, xp, fp)
    # print(intrp)