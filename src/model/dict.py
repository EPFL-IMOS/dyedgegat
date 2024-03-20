import torch.nn as nn

from torch_geometric.nn import (    
    GCN, GAT, GIN,
    GCNConv,
    GATv2Conv,
    GINEConv,
    LayerNorm,
    BatchNorm,
    GraphNorm,
)

RNN_LAYER_DICT = {
    'gru': nn.GRU,
    'lstm': nn.LSTM,
}

ACTIVATION_DICT = {
    'leakyrelu': nn.LeakyReLU,
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'softplus': nn.Softplus,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity,
}

NORM_LAYER_DICT = {
    'layer': LayerNorm,
    'batch': BatchNorm,
    'graph': GraphNorm,
}

GNN_DICT = {
    'gat': GAT,
    'gcn': GCN,
    'gin': GIN,
}

GNN_CONV_LAYER_DICT = {
    'gat': GATv2Conv,
    'gcn': GCNConv,
    'gin': GINEConv,
}
