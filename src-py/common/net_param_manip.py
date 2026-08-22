import torch


def init_xavier_weights(layer):
    """
    https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    :param layer:
    :return:
    """
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def count_params(model, only_trainable):
    """
    :param model: a PyTorch neural network.
    :param only_trainable: If true, only trainable parameters are counted; otherwise total number of parameters are
    counted.
    :return: Total number of parameters of a pytorch neural net.
    """
    if only_trainable:
        params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        params_count = sum(p.numel() for p in model.parameters())
    return params_count


def has_nan(net):
    """
    :return: True if there is Nan among a torch network `net` parameters
    """
    for params in net.parameters():
        if torch.isnan(params).any():
            return True
    return False

