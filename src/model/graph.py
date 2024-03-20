"""
Graph representation learning module

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    softmax,
    sort_edge_index,
    is_undirected,
    to_undirected
)
from torch_geometric.nn.inits import glorot

from ..config import cfg
from ..utils.init import init_weights


class TimeEncode(nn.Module):
    """
    https://github.com/twitter-research/tgn/blob/master/model/time_encoding.py
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class FeatureGraph(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 n_nodes,
                 dropout=0.3,
                 reduce_type='knn',
                 topk=20,
                 learn_sys=True,
                 **kwargs
                ):
        super(FeatureGraph, self).__init__()
        self.learn_sys = learn_sys
        self.n_nodes = n_nodes
        self.in_channels = in_channels
        self.out_channels = embed_dim
        self.lin_l = nn.Linear(in_channels, embed_dim)
        self.lin_r = nn.Linear(in_channels, embed_dim)
        self.att = nn.Parameter(torch.Tensor(1, embed_dim))
        self.dropout = dropout
        self.topk = topk
        if topk == cfg.dataset.n_nodes:
            reduce_type = 'none'
        self.reduce_type = reduce_type
        self.apply(init_weights)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att.data)

    def forward(self, x, edge_index, batch):
        """_summary_

        Args:
            x (Tensor): node with temporal features
            edge_index (Tensor): fully connected feature graph

        Returns:
            _type_: _description_
        """
        b = int(batch[-1] + 1)

        x_l = self.lin_l(x)
        # x_r = x_l
        x_r = self.lin_r(x)

        edge_index, _ = remove_self_loops(edge_index)
        if cfg.model.dyedgegat.add_self_loop:
            edge_index, _ = add_self_loops(edge_index,
                                        num_nodes=x.size(0))

        edge_index = sort_edge_index(edge_index)

        l, r = edge_index
        x_i = x_l[l]
        x_j = x_r[r]
        x = F.leaky_relu(x_i + x_j, negative_slope=0.2)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, l)

        if self.reduce_type == 'knn':
            # perm = topk(alpha, self.topk, l)
            # attention = alpha[perm].view(-1, 1)
            # old_edge_index = edge_index[:, perm]

            alpha = alpha.view(b, self.n_nodes, self.n_nodes) 
            alpha = alpha - torch.diag_embed(torch.diagonal(alpha, dim1=1, dim2=2))
            attention, indices = torch.topk(alpha, self.topk)
            attention = attention.view(-1)

            edge_num = self.topk * self.n_nodes
            index_i = torch.arange(0, self.n_nodes).unsqueeze(1).repeat(1, self.topk).flatten().to(cfg.device).unsqueeze(0)
            index_i = index_i.repeat(b, 1).view(1, -1)
            index_j = indices.clone().detach().view(1, -1)
            for i in range(b):
                index_i[:, i*edge_num:(i+1)*edge_num] += i*self.n_nodes
                index_j[:, i*edge_num:(i+1)*edge_num] += i*self.n_nodes

            new_edge_index = torch.cat((index_i, index_j), dim=0)
            # attention = softmax(attention, new_edge_index[0])

        elif self.reduce_type == 'none':
            attention = alpha.view(-1, 1)
            new_edge_index = edge_index

        # attention = F.dropout(attention, p=self.dropout, training=self.training)
        if self.learn_sys and not is_undirected(new_edge_index):
            undirected_edge_index, undirected_attention = to_undirected(new_edge_index, attention)
            return undirected_edge_index, undirected_attention
        else:
            return new_edge_index, attention


class TemporalGraph(nn.Module):
    def __init__(
        self,
        embed_dim,
        win=5,
        kernel_size=5,
        dropout=0.,
        use_time_encoding=False,
        time_dim=5,
        **kwargs):
        super(TemporalGraph, self).__init__()
        self.win = win
        self.dropout = dropout

        self.cnn = nn.Conv1d(1, 1, kernel_size, padding='same')
        
        in_dim = win + time_dim if use_time_encoding else win
        self.lin_l = nn.Linear(in_dim, embed_dim)
        self.lin_r = nn.Linear(in_dim, embed_dim)
        self.time_embedding = TimeEncode(time_dim) if use_time_encoding else None

        # self.lin = nn.Linear(in_dim, embed_dim)
        self.att = nn.Parameter(torch.Tensor(1, embed_dim))
        self.apply(init_weights)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att.data)
    
    def forward(self, x, edge_index, time=None):
        """_summary_

        Args:
            x (Tensor): node [b, nf, k], nf: # node features, k: seq length
            edge_index (Tensor): fully connected feature graph
            edge_attr (Tensor): relative time step as edge attributes

        Returns:
            _type_: _description_
        """
        edge_index, _ = remove_self_loops(edge_index)
        n_b_nodes = x.shape[0] # b * n_nodes
        
        # x: (b, window_size)
        # smooth x with cnn x: (n, k)
        x = torch.sigmoid(self.cnn(x.unsqueeze(1)).squeeze(1)) # x: (b, smoothed window size)

        x_temp_neigh = x.unfold(dimension=1, size=self.win, step=1)  # x: (b, temporal node size, temporal window size)
        n_temp_nodes = x_temp_neigh.shape[1]
        temporal_edge_index = edge_index.repeat_interleave(n_temp_nodes)
        x_temp_neigh_b = x_temp_neigh.reshape(-1, self.win)
        tmp = torch.add(temporal_edge_index.view(-1, n_temp_nodes).T, torch.arange(0, n_temp_nodes, device=cfg.device).view(-1, 1)).T
        temporal_edge_index_b = torch.add(tmp.T, edge_index.view(-1), alpha=n_temp_nodes-1).T
        temporal_edge_index_b = temporal_edge_index_b.view(2, -1)
        l, r = temporal_edge_index_b

        if self.time_embedding is not None:
            if time is None:
                time = torch.arange(0, n_temp_nodes, device=cfg.device, dtype=torch.float).view(-1, 1) # (seq_len - win + 1, time_dim)
                t_vec = self.time_embedding(time) # (seq_len - win + 1, time_dim)
                x_temp_neigh_b = torch.cat([x_temp_neigh_b, t_vec.repeat([n_b_nodes, 1])], dim=1)
            else:
                seq_len = x.shape[-1]
                time = time.view(-1, seq_len)[:,:n_temp_nodes]
                b = time.shape[0]
                n_nodes = n_b_nodes // b 
                t_vec = self.time_embedding(time)
                x_temp_neigh_b = torch.cat([x_temp_neigh_b, t_vec.repeat([n_nodes, 1])], dim=1)
        
        x_l = self.lin_l(x_temp_neigh_b)
        x_r = self.lin_r(x_temp_neigh_b)
        x_i = x_l[l]
        x_j = x_r[r]

        # Attention weights
        x = F.leaky_relu(x_i + x_j, negative_slope=0.2)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, l)

        new_edge_attr = alpha.view(-1, n_temp_nodes)

        return new_edge_attr
