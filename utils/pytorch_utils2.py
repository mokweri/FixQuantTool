import torch


def build_optimizer(net_params, opt_type, opt_param, init_lr, weight_decay, no_decay_keys):
    if no_decay_keys is not None:
        assert isinstance(net_params, list) and len(net_params) == 2
        net_params = [
            {"params": net_params[0], "weight_decay": weight_decay},
            {"params": net_params[1], "weight_decay": 0},
        ]
    else:
        net_params = [{"params": net_params, "weight_decay": weight_decay}]

    if opt_type == "sgd":
        opt_param = {} if opt_param is None else opt_param
        momentum, nesterov = opt_param.get("momentum", 0.9), opt_param.get(
            "nesterov", True
        )
        optimizer = torch.optim.SGD(
            net_params, init_lr, momentum=momentum, nesterov=nesterov
        )
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(net_params, init_lr)
    else:
        raise NotImplementedError
    return optimizer
