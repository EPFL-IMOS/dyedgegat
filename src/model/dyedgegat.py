import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import (    
    GCN, GAT, GIN,
    GCNConv,
    GATv2Conv,
    GINEConv,
    LayerNorm,
    BatchNorm,
    GraphNorm,
)
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    softmax,
    sort_edge_index,
    is_undirected,
    to_undirected
)

from ..utils.init import init_weights
from ..config import cfg


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


class TimeEncode(torch.nn.Module):

  def __init__(self, dimension):
    """Time encoder module for TGAT model reimplemented by TGN authors.
        https://github.com/twitter-research/tgn/blob/master/model/time_encoding.py
    """
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

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


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(
        self, 
        window_size, 
        in_dim, 
        hid_dim, 
        out_dim, 
        n_layers, 
        dropout,
        do_norm=False,
        act=nn.ReLU(),
        norm_func=BatchNorm,
    ):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, batch_first=True, num_layers=n_layers, dropout=self.dropout)
        self.fc = nn.Linear(hid_dim, out_dim)
        self.act = act
        self.do_norm = do_norm
        if do_norm:
            self.feat_norm = norm_func(hid_dim)
        # self.apply(init_weights)

    def forward(self, x, batch,  c=None):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        if c is not None:
            control_context = c.repeat_interleave(cfg.dataset.n_nodes, 0).unsqueeze(0)  # (1, b*n_nodes, hidden)
            decoder_out, _ = self.gru(h_end_rep, control_context)
        else:
            decoder_out, _ = self.gru(h_end_rep)
        
        if self.do_norm:
            if isinstance(self.feat_norm, GraphNorm):
                decoder_out = self.feat_norm(decoder_out, batch=batch)
            elif isinstance(self.feat_norm, BatchNorm):
                decoder_out = self.feat_norm(decoder_out.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                decoder_out = self.feat_norm(decoder_out)
        decoder_out = self.act(decoder_out)
        out = self.fc(decoder_out)
        return out


ENCODER_DICT = {
    'gru': GRUEncoder,
}


class DyEdgeGAT(nn.Module):
    def __init__(
        self,
        feat_input_node,
        feat_target_node,
        feat_input_edge,
        node_encoder_type='gru',
        node_encoder_mode='univariate',
        contr_encoder_type='gru',
        infer_temporal_edge=True,
        temp_edge_hid_dim=100,
        temp_edge_embed_dim=1,
        temporal_window=5,
        temporal_kernel=5,
        use_time_encoding=True,
        time_dim=5,
        temp_node_embed_dim=16,
        infer_static_graph=True,
        feat_edge_hid_dim=128,
        topk=20,
        learn_sys=True,
        aug_feat_edge_attr=True,
        num_gnn_layers=1,
        gnn_embed_dim=16,
        gnn_type='gin',
        dropout=0.3,
        do_encoder_norm=True,
        do_gnn_norm=True,
        do_decoder_norm=True,
        encoder_norm_type='batch',
        gnn_norm_type='batch',
        decoder_norm_type='batch',
        recon_hidden_dim=10,
        num_recon_layers=1,
        edge_aggr='dot',
        act='relu',
        aug_control=True,
        flip_output=False,
    ):
        super(DyEdgeGAT, self).__init__()
        self.infer_temporal_edge = infer_temporal_edge
        self.infer_graph = infer_static_graph
        self.edge_aggr = edge_aggr
        self.aug_control = aug_control
        self.flip_output = flip_output
        # TODO fix this?
        recon_hidden_dim = temp_node_embed_dim

        if self.aug_control:
            self.control_encoder = ENCODER_DICT[contr_encoder_type](
                in_channels=cfg.dataset.ocvar_dim,
                out_channels=temp_node_embed_dim,
                norm_func=encoder_norm_type if do_encoder_norm != 'none' else None, # type: ignore
                mode='multivariate',
            )

        self.node_encoder = ENCODER_DICT[node_encoder_type](
            in_channels=feat_input_node,
            out_channels=temp_node_embed_dim,
            norm_func=encoder_norm_type if do_encoder_norm != 'none' else None, # type: ignore
            mode=node_encoder_mode,
        )

        if self.infer_graph:
            self.feat_edge_layer = FeatureGraph(
                n_nodes=cfg.dataset.n_nodes,
                in_channels=temp_node_embed_dim,
                embed_dim=feat_edge_hid_dim,
                topk=topk,
                learn_sys=learn_sys,
            )

        self.gnn_edge_dim = 1

        if self.infer_temporal_edge:
            self.edge_encoder = GRUEncoder(
                in_channels=feat_input_edge,
                out_channels=temp_edge_embed_dim,
                norm_func=encoder_norm_type if do_encoder_norm != 'none' else None, # type: ignore
                mode='simple',
            )
            self.temp_edge_layer = TemporalGraph(
                win=temporal_window,
                aug_feat_edge_attr=aug_feat_edge_attr,
                temporal_kernel=temporal_kernel,
                embed_dim=temp_edge_hid_dim,
                use_time_encoding=use_time_encoding,
                time_dim=time_dim,
            )

            self.gnn_edge_dim = temp_edge_embed_dim

            if self.edge_aggr == 'cat':
                self.gnn_edge_dim = temp_edge_embed_dim + feat_input_edge

        self.gnn_layers = ModuleList()
        if do_gnn_norm:
            self.node_norm_layers = ModuleList()
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.do_gnn_norm = do_gnn_norm
        self.act = ACTIVATION_DICT[act]()
        self.dropout = dropout

        gnn_in_channels = temp_node_embed_dim
        for _ in range(num_gnn_layers):
            gnn_out_channels = gnn_embed_dim
            # gnn_out_channels = gnn_embed_dim if i != num_gnn_layers - 1 else head_embed_dim
            if self.gnn_type == 'gin':
                self.gnn_layers.append(GNN_CONV_LAYER_DICT[gnn_type](
                    nn=nn.Sequential(
                        nn.Linear(gnn_in_channels, gnn_embed_dim),
                        nn.BatchNorm1d(gnn_embed_dim),
                        self.act,
                        nn.Linear(gnn_embed_dim, gnn_out_channels),
                        self.act,
                    ),
                    edge_dim=self.gnn_edge_dim,
                ))
            elif self.gnn_type == 'gat':
                self.gnn_layers.append(GNN_CONV_LAYER_DICT[gnn_type](
                    in_channels=gnn_in_channels,
                    out_channels=gnn_out_channels,
                    share_weights=True,
                    edge_dim=self.gnn_edge_dim,
                ))
            else:
                self.gnn_layers.append(GNN_CONV_LAYER_DICT[gnn_type](
                    in_channels=gnn_in_channels,
                    out_channels=gnn_out_channels,
                ))

            if do_gnn_norm:
                self.node_norm_layers.append(NORM_LAYER_DICT[gnn_norm_type](gnn_out_channels))

            gnn_in_channels = gnn_out_channels

        if self.aug_control:
            if gnn_embed_dim != temp_node_embed_dim:
                self.gnn_to_head = torch.nn.Linear(gnn_embed_dim, temp_node_embed_dim)
            else:
                self.gnn_to_head = torch.nn.Identity()

        self.recon = ReconstructionModel(
            window_size=cfg.dataset.window_size,
            in_dim=temp_node_embed_dim if self.aug_control else gnn_embed_dim,
            hid_dim=recon_hidden_dim, # TODO: currently recon_hidden_dim has to be the same as temp_node_embed_dim
            out_dim=feat_target_node,
            n_layers=num_recon_layers,
            do_norm=do_decoder_norm,
            act=self.act,
            norm_func=NORM_LAYER_DICT[decoder_norm_type],
            dropout=dropout
        )
        
    def learn_graph(self, x_temp, batch):
        if self.infer_graph:
            f_edge_index, f_edge_feat = self.feat_edge_layer(
                x_temp,
                edge_index=batch.edge_index,
                batch=batch.batch
            )
        else:
            f_edge_index = batch.edge_index
            f_edge_feat = torch.ones(len(batch.edge_index[0]), device=cfg.device)

        if self.infer_temporal_edge:
            edge_attr = self.temp_edge_layer(
                batch.x,
                edge_index=f_edge_index,
            )

            # temporal edge representation
            edge_attr_temp = self.edge_encoder(edge_attr, batch=batch.batch)
            # edge_attr_temp = edge_attr_temp.view(-1)
            if self.edge_aggr == 'sum':
                aggr_edge_attr =  edge_attr_temp.view(-1, self.gnn_edge_dim) + f_edge_feat.view(-1, 1)
            elif self.edge_aggr == 'dot':
                aggr_edge_attr = f_edge_feat.view(-1, 1) * edge_attr_temp.view(-1, self.gnn_edge_dim)
            elif self.edge_aggr == 'temp':
                aggr_edge_attr = edge_attr_temp
            elif self.edge_aggr == 'cat':
                aggr_edge_attr = torch.cat([f_edge_feat, edge_attr_temp], dim=1)
            else:
                raise ValueError(f"Unknown edge_aggr type: {self.edge_aggr}")

        else:
            aggr_edge_attr = f_edge_feat
        
        return f_edge_index, aggr_edge_attr
        

    def forward(self, batch, return_graph=False):
        # batch.x [b, nf, k], b: batch size, nf: number of features, k: sequence length

        c_temp = None
        if self.aug_control:
            c_temp = self.control_encoder(batch.c, batch=batch.batch)

        # temporal node representation
        x_temp = self.node_encoder(batch.x, batch=batch.batch, c=c_temp)
        
        f_edge_index, aggr_edge_attr = self.learn_graph(x_temp, batch)

        x = x_temp
        for i in range(self.num_gnn_layers):
            x0 = x

            gnn_attr = dict(
                x=x,
                edge_index=f_edge_index,
            )
            # 'gat' only takes edge attr
            if self.gnn_type == 'gat' or self.gnn_type == 'gin':
                gnn_attr['edge_attr'] = aggr_edge_attr.view(-1, self.gnn_edge_dim)
            # 'gcn' only takes edge weight with dim=1
            elif self.gnn_type == 'gcn':
                gnn_attr['edge_weight'] = aggr_edge_attr

            x = self.gnn_layers[i](**gnn_attr)

            if self.do_gnn_norm:
                if isinstance(self.node_norm_layers[i], GraphNorm):
                    x = self.node_norm_layers[i](x, batch=batch.batch)
                else:
                    x = self.node_norm_layers[i](x)

            if i!= 0 and i != self.num_gnn_layers-1:
                x = x + x0

            x = self.act(x)

            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.aug_control:
            x = self.gnn_to_head(x)

        c_temp_reverse = None
        if self.aug_control:
            c_reverse = torch.flip(batch.c, [1])
            c_temp_reverse = self.control_encoder(c_reverse, batch=batch.batch)
        
        recon = self.recon(x, batch=batch.batch, c=c_temp_reverse)
        
        if self.flip_output:
            # reverse the sequence back
            recon = torch.flip(recon, [1])

        if return_graph:
            return recon, f_edge_index, aggr_edge_attr
        return recon