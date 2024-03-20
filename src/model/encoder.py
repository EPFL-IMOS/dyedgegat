
# for binary inputs
import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import GraphNorm

from ..model.dict import (
    ACTIVATION_DICT, 
    NORM_LAYER_DICT
)
from ..config import cfg
from ..utils.init import init_weights


class EmbeddingEncoder(nn.Module):
    def __init__(
            self,
            in_channels, 
            out_channels,
            activation='relu',
            **kwargs
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.act = ACTIVATION_DICT[activation]()
        self.mlp = nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        nn.BatchNorm1d(out_channels),
                        self.act,
                        nn.Linear(out_channels, out_channels),
                        self.act,
                    )
        # self.embed = nn.Embedding(n_classes, out_channels)
        # torch.nn.init.xavier_uniform_(self.embed.weight.data)
        self.apply(init_weights)

    def forward(self, x, **kwargs):
        seq = x.shape[1]
        x = x.reshape(-1, self.in_channels, seq) # (b, nf, seq)
        # Encode just the first dimension if more exist
        x = x[:, :, 0] # (b, nf)

        x = self.mlp(x) # (b, 1)
        # x = self.embed(x.long()) # (b, 1, out_channels)
        # x = x.squeeze(1) # (b, out_channels)

        return x


class GRUEncoder(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation='relu',
        norm_func='batch',
        bidirectional=False,
        mode='univariate',
        **kwargs
    ):
        super(GRUEncoder, self).__init__()
        self.mode = mode
        self.act = ACTIVATION_DICT[activation]()
        self.do_norm = norm_func is not None
        self.in_channels = in_channels

        if mode == 'univariate':
            self.grus = ModuleList([
                nn.GRU(
                    in_channels,
                    out_channels,
                    bidirectional=bidirectional,
                    batch_first=True) for _ in range(cfg.dataset.n_nodes)
            ])
            if self.do_norm:
                self.feat_norms = ModuleList([
                    NORM_LAYER_DICT[norm_func](out_channels) for _ in range(cfg.dataset.n_nodes)
                ])
        else:
            self.gru = nn.GRU(
                in_channels,
                out_channels,
                bidirectional=bidirectional,
                batch_first=True
            )
            if self.do_norm:
                self.feat_norm = NORM_LAYER_DICT[norm_func](out_channels)
        self.apply(init_weights)

    def forward(self, x, batch, c=None, return_all=False):
        b = batch[-1] + 1

        if self.mode == 'simple':
            x = x.unsqueeze(2)  # (b, k, nf)
        elif self.mode == 'univariate' or self.mode == 'multivariate':
            seq = x.shape[1]
            x = x.reshape(b, -1, seq)
            x = x.permute(0, 2, 1)  # (b, seq, nf)
        
        if self.mode == 'univariate':
            hs = []
            outs = []
            for i in range(cfg.dataset.n_nodes):
                xi = x[:, :, (i,)]
                if c is not None and c.dim() == 2:
                    c = c.unsqueeze(0)
                out_i, hi = self.grus[i](xi, c)
                out_i, hi = out_i[:, -1, :], hi[-1, :, :]  # Extracting from last layer # (b, n, k)
                
                if self.do_norm:
                    if isinstance(self.feat_norms[i], GraphNorm):
                        hi = self.feat_norms[i](hi, batch=batch)
                    else:
                        hi = self.feat_norms[i](hi)
                hs.append(hi)
                outs.append(out_i)
            h = torch.stack(hs, dim=1)
            out = torch.stack(outs, dim=1)
            h = h.reshape(b * cfg.dataset.n_nodes, -1)
            out = out.reshape(b * cfg.dataset.n_nodes, -1)
        else:
            if c is not None:
                if self.mode == 'multivariate':
                    h0 = c.unsqueeze(0)  # (1, b, hidden)
                else:
                    h0 = c.repeat_interleave(cfg.dataset.n_nodes, 0).unsqueeze(0)  # (1, b*n_nodes, hidden)
                out, h = self.gru(x, h0)
            else:
                out, h = self.gru(x)
            out, h = out[:, -1, :], h[-1, :, :]  # Extracting from last layer # (b, n, k)
        
            if self.do_norm:
                if isinstance(self.feat_norm, GraphNorm):
                    h = self.feat_norm(h, batch=batch)
                else:
                    h = self.feat_norm(h)
        
        h = self.act(h)
        
        if return_all:
            return out, h
        return h


ENCODER_DICT = {
    'gru': GRUEncoder,
    'embedding': EmbeddingEncoder,
}
