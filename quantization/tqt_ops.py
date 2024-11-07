import torch
import numpy as np
import math
import torch.nn.functional as F


def kl_divergence(ha, hb):
    """
        ans = J_{kl}(a,b)
    """
    return torch.sum(ha * torch.log(ha / hb))


def quantize_bins_and_expand(dist, quant_bins):
    dist_len = dist.shape[0]
    width = math.floor(1. * dist_len / quant_bins)
    dist_q = torch.zeros([quant_bins])
    dist_e = 0. * dist
    for i in range(quant_bins):
        if i != quant_bins - 1:
            dist_q[i] = dist[i * width:(i + 1) * width].sum()
            width_preserve = width - (dist[i * width:(i + 1) * width]== 0).sum()
            if width_preserve == 0:
                dist_e[i * width:(i + 1) * width] = 0
            else:
                dist_e[i * width:(i + 1) * width] = dist_q[i] / width_preserve
        else:
            dist_q[i] = dist[i * width:].sum()
            width_preserve = width - (dist[i * width:] == 0).sum()
            if width_preserve == 0:
                dist_e[i * width:] = 0
            else:
                dist_e[i * width:] = dist_q[i] / width_preserve
    return dist_e * (dist != 0)


def _calc_threshold(x, bin_number=2048, cali_number=128, eps=1e-12):
    q = x.flatten().data
    dist = torch.histc(q, bins=bin_number) + eps
    bin_width = (q.max() - q.min()) / bin_number
    divergence = torch.zeros([bin_number]) * 1.0

    for i in range(cali_number, bin_number):
        ref_dist = dist[:i].clone()
        outliers_count = dist[i:].sum()
        ref_dist[-1] += outliers_count
        ref_dist /= ref_dist.sum()
        can_dist = quantize_bins_and_expand(dist[:i], cali_number)
        can_dist /= can_dist.sum()
        divergence[i] = kl_divergence(ref_dist, can_dist)
    m, m_idx = torch.min(divergence[cali_number:], 0)
    threshold = q.min() + (m_idx + cali_number + 0.5) * bin_width + eps
    log2_t = torch.tensor([torch.log2(threshold)])
    return log2_t


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


if __name__ == '__main__':

    # Example usage for testing
    class Example:
        def __init__(self, bitwidth=8, tensor_type='act'):
            self.bitwidth = torch.tensor([bitwidth])
            self.tensor_type = tensor_type

        def init_threshold(self, x):
            th = _init_threshold(x)
            return np.log2(th)

    class Example2:
        def __init__(self, bitwidth=8, tensor_type='act'):
            self.bitwidth = torch.tensor([bitwidth])
            self.tensor_type = tensor_type

        def init_threshold(self, x):
            return _calc_threshold(x)

    # Test setup
    example = Example(bitwidth=8, tensor_type='act')
    example2 = Example2(bitwidth=8, tensor_type='act')
    x = np.random.randn(2000).astype(np.float32)
    y = torch.from_numpy(x)
    y.cuda()

    # Calculate threshold in PyTorch
    threshold_numpy = example.init_threshold(x)
    print("Threshold (Numpy):", threshold_numpy)

    threshold_torch = example2.init_threshold(y)
    print("Threshold (Torch):", threshold_torch)
