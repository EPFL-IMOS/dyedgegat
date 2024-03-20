import torch.nn as nn
from torch_geometric.nn import (
    LayerNorm,
    GraphNorm
)

from ..config import cfg

# TODO: check what initialization is better for which module

def init_weights(m):
    r"""
    Performs weight initialization
    Args:
        m (nn.Module): PyTorch module
    """
    gain_act = cfg.model.activation if cfg.model.activation in ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'selu'] else 'relu'
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, LayerNorm) or isinstance(m, GraphNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.weight.data = nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain(gain_act))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Embedding):
        m.weight.data = nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain(gain_act))
    elif isinstance(m, nn.Parameter):
        m.data = nn.init.xavier_uniform_(gain=nn.init.calculate_gain(gain_act))
    elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)