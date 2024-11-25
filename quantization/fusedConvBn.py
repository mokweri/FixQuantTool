import torch
import torch.nn as nn
from torch.nn import functional as F

from quantization.emu_modules import FxP_QConv2D
from quantization.qmodules import QuantizedConv2d
from quantization.tqt import TQTQuantizer

# Number of steps before freezing the batch norm running average and variance
# change if you cahnge dataset
FREEZE_BN_DELAY_DEFAULT = 93200 #200000


_conv_meta = {'conv1d': (1, F.conv1d),
              'conv2d': (2, F.conv2d),
              'conv3d': (3, F.conv3d)}


class FusedConvBN(nn.Module):
    def __init__(self, conv_mod, bn_mod, freeze_bn_delay=FREEZE_BN_DELAY_DEFAULT, _mod_name=None):
        if not bn_mod.track_running_stats:
            raise ValueError("FusedConv BN folding is only supported for BatchNorm which tracks running stats")

        super(FusedConvBN, self).__init__()
        self.conv_mod = conv_mod
        self.bn_mod = bn_mod
        self.freeze_bn_delay = freeze_bn_delay
        self.frozen = False

        self.weight_quantizer = TQTQuantizer(bitwidth=8, tensor_type='weight')
        self.bias_quantizer = TQTQuantizer(bitwidth=8, tensor_type='weight')
        self.act_quantizer = TQTQuantizer(bitwidth=8, tensor_type='act')

        self._mod_name = None
        if isinstance(conv_mod, nn.Linear):
            self.conv_forward_fn = self._linear_layer_forward
            self.conv_module_type = "fc"
        elif isinstance(conv_mod, nn.Conv1d):
            self.conv_forward_fn = self._conv_layer_forward
            self.conv_module_type = "conv1d"
        elif isinstance(conv_mod, nn.Conv2d):
            self.conv_forward_fn = self._conv_layer_forward
            self.conv_module_type = "conv2d"
        else:
            self.conv_forward_fn = self._conv_layer_forward
            self.conv_module_type = "conv3d"

    @staticmethod
    def verify_module_types(param_module, bn):
        foldable_seqs = [((nn.Linear, nn.Conv1d), nn.BatchNorm1d),
                         (nn.Conv2d, nn.BatchNorm2d),
                         (nn.Conv3d, nn.BatchNorm3d)]
        error_msg = "Can't fold sequence of {} --> {}. ".format(param_module.__class__.__name__, bn.__class__.__name__)
        for seq in foldable_seqs:
            if isinstance(param_module, seq[0]):
                if not isinstance(bn, seq[1]):
                    raise TypeError(error_msg + "{} must be followed by {}".
                                    format(param_module.__class__.__name__, seq[1].__name__))
                return
        raise TypeError(error_msg + "Only Conv/Linear modules followed by BatchNorm modules allowed"
                        .format(param_module.__class__.__name__, bn.__class__.__name__))

    @property
    def module_name(self):
        return self._mod_name

    @module_name.setter
    def module_name(self, value):
        self._mod_name = value

    def forward(self, x):
        """
        According to https://arxiv.org/pdf/1806.08342.pdf section 3.2.2.
        Note:
            The conv layer bias doesn't get included in the calculation!
            When calculating the batch norm, the bias offsets the mean and so when calculating (x - mu)
            we get the unbiased position w.r.t. to the mean.
            i.e. the result of the forward is:
            bn(conv(x)) = ( conv(x) - E(conv(x)) ) * gamma / std(conv(x)) + beta =
                          = ( x*W + B - E(x*W +B) ) * gamma / sqrt(E((x*W+ B - E(x*W +B))^2)) + beta =
                          = (x*W -E(x*W)) * gamma / std(x*W) + beta
        """
        if not self.frozen:
            w, b, gamma, beta = self._get_all_parameters()
            if self.training:
                # -------     1st forward pass
                batch_mean, batch_var = self.get_batch_stats(self.conv_forward_fn(x, w), b)
                recip_sigma_batch = torch.rsqrt(batch_var + self.bn_mod.eps)
                with torch.no_grad():
                    sigma_running = torch.sqrt(self.bn_mod.running_var + self.bn_mod.eps)
                w_corrected = w * self.broadcast_correction_weight(gamma / sigma_running)

                # -------     2nd forward pass
                w_quantized = self.weight_quantizer.forward(w_corrected) # weight Quantizer
                recip_c = self.broadcast_correction(sigma_running * recip_sigma_batch)
                bias_corrected = beta - gamma * batch_mean * recip_sigma_batch
                bias_quantized = self.broadcast_correction(self.bias_quantizer.forward(bias_corrected)) # bias quantizer
                y = self.conv_forward_fn(x, w_quantized, None)     # 2nd forward call
                y.mul_(recip_c).add_(bias_quantized)
            else:
                with torch.no_grad():
                    recip_sigma_running = torch.rsqrt(self.bn_mod.running_var + self.bn_mod.eps)
                w_corrected = w * self.broadcast_correction_weight(gamma * recip_sigma_running)
                w_quantized = self.weight_quantizer.forward(w_corrected) # weight Quantizer
                corrected_mean = self.bn_mod.running_mean - (b if b is not None else 0)
                bias_corrected = beta - gamma * corrected_mean * recip_sigma_running
                bias_quantized = self.bias_quantizer.forward(bias_corrected)  # biasQuantizer
                y = self.conv_forward_fn(x, w_quantized, bias_quantized)
        else:
            w, b = self.conv_mod.weight, self.conv_mod.bias
            w_quantized = self.weight_quantizer.forward(w)
            bias_quantized = self.bias_quantizer.forward(b)
            y = self.conv_forward_fn(x, w_quantized, bias_quantized)

        y = self.act_quantizer.forward(y) # quantize the activation
        return y

    def broadcast_correction(self, c: torch.Tensor):
        """
        Broadcasts a correction factor to the output for elementwise operations.
        """
        expected_output_dim = 2 if self.conv_module_type == "fc" else _conv_meta[self.conv_module_type][0] + 2
        view_fillers_dim = expected_output_dim - c.dim() - 1
        view_filler = (1,) * view_fillers_dim
        expected_view_shape = c.shape + view_filler
        return c.view(*expected_view_shape)

    def broadcast_correction_weight(self, c: torch.Tensor):
        """
        Broadcasts a correction factor to the weight.
        """
        if c.dim() != 1:
            raise ValueError("Correction factor needs to have a single dimension")
        expected_weight_dim = 2 if self.conv_module_type == "fc" else _conv_meta[self.conv_module_type][0] + 2
        view_fillers_dim = expected_weight_dim - c.dim()
        view_filler = (1,) * view_fillers_dim
        expected_view_shape = c.shape + view_filler
        return c.view(*expected_view_shape)

    def get_batch_stats(self, x, bias=None):
        """
        Get the batch mean and variance of x and updates the BatchNorm's running mean and average.
        Args:
            x (torch.Tensor): input batch.
            bias (torch.Tensor): the bias that is to be applied to the batch.
        Returns:
            (mean,variance)
        Note:
            In case of `nn.Linear`, x may be of shape (N, C, L) or (N, L) - N is batch size, C is no. of channels, L is the features size.
            The batch norm computes the stats over C in the first case or L on the second case.
            The batch normalization layer is
            (`nn.BatchNorm1d`)[https://pytorch.org/docs/stable/nn.html#batchnorm1d]

            In case of `nn.Conv2d`, x is of shape (N, C, H, W)
            where H,W are the image dimensions, and the batch norm computes the stats over C.
            The batch normalization layer is
            (`nn.BatchNorm2d`)[https://pytorch.org/docs/stable/nn.html#batchnorm2d]
        """
        channel_size = self.bn_mod.num_features
        self.bn_mod.num_batches_tracked += 1

        # Calculate current batch stats
        batch_mean = x.transpose(0, 1).contiguous().view(channel_size, -1).mean(1)
        # BatchNorm currently uses biased variance (without Bessel's correction) as was discussed at
        # https://github.com/pytorch/pytorch/issues/1410
        # also see the source code itself:
        # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L216
        batch_var = x.transpose(0, 1).contiguous().view(channel_size, -1).var(1, unbiased=False)

        # Update running stats
        with torch.no_grad():
            biased_batch_mean = batch_mean + (bias if bias is not None else 0)
            # However - running_var is updated using unbiased variance!
            # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L223
            n = x.numel() / channel_size
            corrected_var = batch_var * (n / (n - 1))
            momentum = self.bn_mod.momentum
            if momentum is None:
                # momentum is None - we compute a cumulative moving average
                # as noted in https://pytorch.org/docs/stable/nn.html#batchnorm2d
                momentum = 1. / float(self.bn_mod.num_batches_tracked)
            self.bn_mod.running_mean.mul_(1 - momentum).add_(momentum * biased_batch_mean)
            self.bn_mod.running_var.mul_(1 - momentum).add_(momentum * corrected_var)
        # print(self.bn_mod.num_batches_tracked)
        if self.bn_mod.num_batches_tracked > self.freeze_bn_delay:
            self.freeze()

        return batch_mean, batch_var

    def _conv_layer_forward(self, input, w, b=None):
        # We implement according to Conv1/2/3d.forward(), but plug in our weights
        conv = self.conv_mod
        ndims, func = _conv_meta[self.conv_module_type]

        # 'circular' padding doesn't exist pre-pytorch 1.1.0
        if getattr(conv, 'padding_mode', None) == 'circular':
            expanded_padding = []
            for pad_idx in reversed(range(ndims)):
                expanded_padding.extend([(conv.padding[pad_idx] + 1) // 2, conv.padding[pad_idx] // 2])
            return func(F.pad(input, expanded_padding, mode='circular'),
                        w, b, conv.stride,
                        (0,) * ndims, conv.dilation, conv.groups)
        return func(input, w, b, conv.stride, conv.padding, conv.dilation, conv.groups)

    def freeze(self):
        print("Freezing the BN")
        w, b, gamma, beta = self._get_all_parameters()
        with torch.no_grad():
            recip_sigma_running = torch.rsqrt(self.bn_mod.running_var + self.bn_mod.eps)
            # w.mul_(self.broadcast_correction_weight(gamma * recip_sigma_running))

            w = self.conv_mod.weight.detach().clone()  # Detach to avoid in-place modification error
            w *= self.broadcast_correction_weight(gamma * recip_sigma_running)  # Modify weight
            self.conv_mod.weight = nn.Parameter(w)

            corrected_mean = self.bn_mod.running_mean - (b if b is not None else 0)
            bias_corrected = beta - gamma * corrected_mean * recip_sigma_running
            if b is not None:
                b.copy_(bias_corrected)
            else:
                self.conv_mod.bias = nn.Parameter(bias_corrected)
        self.frozen = True


    def _get_all_parameters(self):
        w, b, gamma, beta = self.conv_mod.weight, self.conv_mod.bias, self.bn_mod.weight, self.bn_mod.bias
        if not self.bn_mod.affine:
            gamma = 1.
            beta = 0.
        return w, b, gamma, beta

    @classmethod
    def from_float(cls, conv, bn):
        fused_convbn = cls(conv, bn, freeze_bn_delay=FREEZE_BN_DELAY_DEFAULT)
        return fused_convbn

    def state_dict(self, *args, prefix='', **kwargs):
        state = super().state_dict(*args, prefix=prefix, **kwargs)
        # Add 'frozen' attribute to the state_dict
        state[prefix+'frozen'] = self.frozen
        return state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Load conv_mod and bn_mod individually
        conv_prefix = prefix + 'conv_mod.'
        bn_prefix = prefix + 'bn_mod.'

        # Handle additional keys if necessary
        frozen_key = prefix + 'frozen'
        if frozen_key in state_dict:
            self.frozen = state_dict[frozen_key]
        state_dict.pop(frozen_key, None)

        # Extract conv_mod parameters
        conv_state_dict = {k[len(conv_prefix):]: v for k, v in state_dict.items() if k.startswith(conv_prefix)}
        self.conv_mod._load_from_state_dict(conv_state_dict, '', local_metadata, strict, missing_keys, unexpected_keys,
                                            error_msgs)

        # Extract bn_mod parameters
        bn_state_dict = {k[len(bn_prefix):]: v for k, v in state_dict.items() if k.startswith(bn_prefix)}
        self.bn_mod._load_from_state_dict(bn_state_dict, '', local_metadata, strict, missing_keys, unexpected_keys,
                                          error_msgs)

        super(FusedConvBN, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                       unexpected_keys, error_msgs)

    def export_quant_info(self):
        # "conv1": {"weight": 8, "bias": 7, "in": 5, "out": 6}, # @TODO Format accordingly
        frac_w = self.weight_quantizer.export_quant_info()[1]
        frac_b = self.bias_quantizer.export_quant_info()[1]
        frac_out = self.act_quantizer.export_quant_info()[1]
        return frac_w, frac_b, frac_out

    def __repr__(self):
        return f'QFusedConvBN({self.conv_mod.__repr__()}, Quantizer=TQT)'

    def to_qconv(self):
        assert self.frozen, 'The BN module is not frozen'
        conv = QuantizedConv2d(
            in_channels=self.conv_mod.in_channels,
            out_channels=self.conv_mod.out_channels,
            kernel_size=self.conv_mod.kernel_size,
            stride=self.conv_mod.stride,
            padding=self.conv_mod.padding,
            dilation=self.conv_mod.dilation,
            groups=self.conv_mod.groups,
            bias=self.conv_mod.bias is not None,
            padding_mode=self.conv_mod.padding_mode,
        )
        conv.weight_quantizer = self.weight_quantizer
        conv.bias_quantizer = self.bias_quantizer
        conv.act_quantizer = self.act_quantizer
        conv._mod_name = self._mod_name

        conv.weight = torch.nn.Parameter(self.conv_mod.weight.detach())
        conv.bias = torch.nn.Parameter(self.conv_mod.bias.detach())

        return conv



if __name__ == '__main__':
    # TEST THE CLASS
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import torch.optim as optim

    # Set up input data and define parameters
    batch_size, in_channels, height, width = 8, 3, 32, 32
    input_data = torch.randn(batch_size, in_channels, height, width).to(device)

    # Conv and BatchNorm configuration
    out_channels = 16
    kernel_size = 3
    stride = 1
    padding = 1

    # Define standard Conv + BatchNorm model
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True).to(device)
    bn = nn.BatchNorm2d(out_channels).to(device)
    standard_model = nn.Sequential(conv, bn)
    standard_model.eval()

    # Define FusedConvBN model
    fused_model = FusedConvBN(conv_mod=conv, bn_mod=bn, freeze_bn_delay=2000).to(device)
    fused_model.eval()

    # Check for forward pass and compare outputs
    with torch.no_grad():
        standard_output = standard_model(input_data)
        fused_output = fused_model(input_data)

    # Compare outputs for initial consistency
    print("Initial output mse:", nn.functional.mse_loss(standard_output, fused_output))
    print(fused_model.export_quant_info())

    # Test freezing behavior
    fused_model.freeze()
    with torch.no_grad():
        fused_frozen_output = fused_model(input_data)

    print("mse after freezing:", nn.functional.mse_loss(standard_output, fused_frozen_output))
    print(fused_model.export_quant_info())

    print("--------------")
    new_conv = fused_model.to_conv()
    # print(new_conv)
    with torch.no_grad():
        newconv_output = new_conv(input_data)

    print("mse after new conv:", nn.functional.mse_loss(standard_output, newconv_output))
    print(new_conv.export_quant_info())



    # torch.save(fused_model.state_dict(), 'fused_conv_bn_model_model.pt')
    # fused_model.load_state_dict(torch.load('fused_conv_bn_model.pt'))

    #print quantizer parameters
    # def quantizer_parameters(model):
    #     return [
    #         param for name, param in model.named_parameters()
    #         if 'log_threshold' in name
    #     ]
    #
    # def non_quantizer_parameters(model):
    #     return [
    #         param for name, param in model.named_parameters()
    #         if 'log_threshold' not in name
    #     ]
    #
    # for name, param in fused_model.named_parameters():
    #     if 'log_threshold' in name:
    #         print(name)

    # # Check if frozen parameters remain constant
    # optimizer = optim.SGD(fused_model.parameters(), lr=1e-3, momentum=0.9)
    # fused_model.train()
    # for _ in range(5):  # Small number of training steps
    #     optimizer.zero_grad()
    #     output = fused_model(input_data)
    #     loss = output.mean()  # Arbitrary loss function for testing
    #     loss.backward()
    #     optimizer.step()
    #
    # # Verify if frozen batchnorm parameters are unaffected by updates
    # with torch.no_grad():
    #     final_output = fused_model(input_data)
    #
    # print("Output difference after training steps:", torch.norm(fused_frozen_output - final_output).item())