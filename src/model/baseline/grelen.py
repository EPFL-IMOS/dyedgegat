"""
Implementation of the Multivariate Time Series Anomaly Detection from the Perspective of Graph Relational Learning (GRELEN)
https://github.com/Vicky-51/GRELEN
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ...config import cfg
from ...train.train import ReconstructionTrainer
from ...metric.loss import kl_categorical, nll_gaussian


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, 
                 n_in, 
                 n_hid, 
                 n_out, 
                 do_prob=0.):
        super(MLP, self).__init__()
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float().to(cfg.device)
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(shape=logits.size(), eps=eps)
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        tau: non-negative scalar temperature
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits=logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(cfg.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0).to(cfg.device)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class Graph_learner(nn.Module):
    def __init__(self, n_in, n_hid, n_head_dim, head, dropout_prob=0.):  # n_in = T
        super(Graph_learner, self).__init__()
        self.n_hid = n_hid
        self.head = head
        self.n_in = n_in
        self.n_head_dim = n_head_dim

        self.mlp1 = MLP(n_in, n_hid, n_hid, dropout_prob)
        self.Wq = nn.Linear(n_hid, n_head_dim * head)
        self.Wk = nn.Linear(n_hid, n_head_dim * head)
        for m in [self.Wq, self.Wk]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs):  # inputs: [B, N, T(features)]
        X = self.mlp1(inputs)
        Xq = self.Wq(X)  # [B, N, n_hid_subspace]
        Xk = self.Wk(X)
        B, N, n_hid = Xq.shape
        Xq = Xq.view(B, N, self.head, self.n_head_dim)  # [B, N, head, head_dim]
        Xk = Xk.view(B, N, self.head, self.n_head_dim)
        Xq = Xq.permute(0, 2, 1, 3)  # [B, head, N, head_dim]
        Xk = Xk.permute(0, 2, 1, 3)
        probs = torch.matmul(Xq, Xk.transpose(-1, -2))  # [B, head, N, N]

        return probs


class DCGRUCell_(torch.nn.Module):
    def __init__(self, num_units: int, max_diffusion_step: int, num_nodes: int, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        """
        Adapted from Pytorch implementation of DCGRU Cell
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru

        self._gconv_0 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units * 2)
        self._gconv_1 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units * 2)
        self._gconv_c_0 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units)
        self._gconv_c_1 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=cfg.device)
        return L

    def _calculate_random_walk_matrix(self, adj_mx):

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(cfg.device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(cfg.device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def _calculate_random_walk0(self, adj_mx, B):  # adj_mx是tensor形式
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).repeat(B, 1, 1).to(cfg.device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(cfg.device), d_inv)
        d_mat_inv = torch.diag_embed(d_inv)
        random_walk_mx = torch.matmul(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, inputs, hx, adj):
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, adj, hx, output_size, bias_start=1.0))

        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv_c(inputs, adj, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        B = inputs.shape[0]
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state  # [B, N, 2 * C]
        x0_0 = torch.unsqueeze(x, 0)
        x1_0 = torch.unsqueeze(x, 0)
        # print('x0_0, shape', x0_0.shape)
        #         x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        #         x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        #         x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            # print('adj_mx_in,', adj_mx.shape)
            #             print('x0,', x0.shape)
            # print('x', x.shape)

            x0_1 = torch.matmul(adj_mx0, x0_0)
            x1_1 = torch.matmul(adj_mx1, x1_0)
            # print('x0_1', x0_1.shape)

            x0_0 = torch.cat([x0_0, x0_1], dim=0)
            x1_0 = torch.cat([x1_0, x1_1], dim=0)

            for k in range(2, self._max_diffusion_step + 1):
                x0_2 = torch.matmul(adj_mx0, x0_1)
                x1_2 = torch.matmul(adj_mx1, x1_1)
                # print('x0_2,', x0_2.shape)
                # print('x0_0,', x0_0.shape)
                x0_0 = torch.cat([x0_0, x0_1], dim=0)
                x1_0 = torch.cat([x1_0, x1_1], dim=0)
                x0_1 = x0_2
                x1_1 = x1_2
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.
        # print([num_matrices, self._num_nodes, input_size, batch_size])
        # print('x0_0', x0_0.shape) # torch.Size([3, 32, 90, 128])
        x0_0 = x0_0.permute(1, 2, 3, 0)  # [3, 90, 128]
        x1_0 = x1_0.permute(1, 2, 3, 0)  # [3, 90, 128]
        #         x = torch.reshape(x0_0, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        #         x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        # weights0 = self._gconv_params0.get_weights((input_size * num_matrices, output_size))
        # x0_0 = torch.matmul(x0_0, weights0)  # (batch_size * self._num_nodes, output_size)
        # biases0 = self._gconv_params0.get_biases(output_size, bias_start)
        # x0_0 += biases0
        x0_0 = self._gconv_0(x0_0)
        x1_0 = self._gconv_1(x1_0)

        # weights1 = self._gconv_params1.get_weights((input_size * num_matrices, output_size))
        # x1_0 = torch.matmul(x1_0, weights1)  # (batch_size * self._num_nodes, output_size)
        # biases1 = self._gconv_params1.get_biases(output_size, bias_start)
        # x1_0 += biases1
        # # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        # # print([batch_size, self._num_nodes * output_size])

        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_nodes * output_size])


    def _gconv_c(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        B = inputs.shape[0]
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state  # [B, N, 2 * C]
        x0_0 = torch.unsqueeze(x, 0)
        x1_0 = torch.unsqueeze(x, 0)


        if self._max_diffusion_step == 0:
            pass
        else:
            x0_1 = torch.matmul(adj_mx0, x0_0)
            x1_1 = torch.matmul(adj_mx1, x1_0)

            x0_0 = torch.cat([x0_0, x0_1], dim=0)
            x1_0 = torch.cat([x1_0, x1_1], dim=0)

            for k in range(2, self._max_diffusion_step + 1):
                x0_2 = torch.matmul(adj_mx0, x0_1)
                x1_2 = torch.matmul(adj_mx1, x1_1)
                x0_0 = torch.cat([x0_0, x0_1], dim=0)
                x1_0 = torch.cat([x1_0, x1_1], dim=0)
                x0_1 = x0_2
                x1_1 = x1_2
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.

        x0_0 = x0_0.permute(1, 2, 3, 0)
        x1_0 = x1_0.permute(1, 2, 3, 0)

        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x0_0 = self._gconv_c_0(x0_0)
        x1_0 = self._gconv_c_1(x1_0)


        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_nodes * output_size])

class EncoderModel(nn.Module):
    def __init__(self, 
        n_dim:int, 
        n_hid:int, 
        max_diffusion_step:int, 
        num_nodes:int, 
        num_rnn_layers:int, 
        filter_type
    ):
        super(EncoderModel, self).__init__()
        self.input_dim = n_dim
        self.rnn_units = n_hid
        self.max_diffusion_step = max_diffusion_step
        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.filter_type = filter_type
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell_(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):

        batch_size = inputs.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size)).to(cfg.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)


class GRELEN(nn.Module):
    """
    GRELEN Model.
    """

    def __init__(
            self, 
            input_dim, 
            window_size, 
            graph_learner_n_hid, 
            graph_learner_n_head_dim,
            graph_learner_head, 
            gru_n_dim, 
            dropout_prob=0.,
            num_rnn_layers=1, 
            max_diffusion_step=2, 
            temperature=0.5,
            filter_type='random', 
            hard=True,
            **kwargs,
        ):  # n_in = T
        super(GRELEN, self).__init__()
        self.log_prior = torch.FloatTensor(np.log(cfg.model.grelen.prior)).unsqueeze(0).unsqueeze(0).to(device=cfg.device) # non trainable prior
        
        self.window_size = window_size
        self.output_seq_len = window_size - 1
        self.graph_learner = Graph_learner(window_size, graph_learner_n_hid, graph_learner_n_head_dim, graph_learner_head,
                                           dropout_prob)
        self.linear1 = nn.Linear(1, gru_n_dim)  # First layer of projection
        nn.init.xavier_normal_(self.linear1.weight.data)
        self.linear1.bias.data.fill_(0.1)

        self.temperature = temperature
        self.hard = hard
        self.gru_n_dim = gru_n_dim
        self.num_nodes = input_dim
        self.head = graph_learner_head
        self.encoder_model = nn.ModuleList(
            [EncoderModel(gru_n_dim, gru_n_dim, max_diffusion_step, self.num_nodes, num_rnn_layers, filter_type) \
             for _ in range(self.head - 1)])
        self.linear_out = nn.Linear(gru_n_dim, 1)
        nn.init.xavier_normal_(self.linear_out.weight.data)
        self.linear_out.bias.data.fill_(0.1)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def reshape_edges(self, edges):
        # edges: B, N(N-1), type
        adj_mat = torch.zeros(edges.shape[-1], edges.shape[0], self.num_nodes, self.num_nodes, device=cfg.device) +1
        mask = ~torch.eye(self.num_nodes, dtype=bool, device=cfg.device).unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(edges.shape[-1], edges.shape[0], 1, 1)
        adj_mat[mask] = edges.permute(2, 0, 1).flatten() # (type, B, N, N)
        return adj_mat

    def encoder(self, inputs, adj, head):
        """
        Encoder forward pass
        """

        encoder_hidden_state = None
        encoder_hidden_state_tensor = torch.zeros(inputs.shape).to(cfg.device)
        for t in range(self.window_size):
            _, encoder_hidden_state = self.encoder_model[head](inputs[..., t], adj, encoder_hidden_state)
            encoder_hidden_state_tensor[..., t] = encoder_hidden_state[-1, ...].reshape(-1, self.num_nodes,
                                                                                        self.gru_n_dim)
        return encoder_hidden_state_tensor

    def forward(self, batch, **kwargs):

        if not isinstance(batch, torch.Tensor):
            x = batch.x  # (b * n_nodes, window_size)
            # Change input shape from (b * n_nodes, window_size) to (b, n_nodes, window_size)
            x = x.view(-1, self.num_nodes, x.shape[1])  # (b, n_nodes, window_size)
        else:
            x = batch

        B = x.shape[0]

        input_projected = self.linear1(x.unsqueeze(-1))  # [B, N, T, GRU_n_dim]
        input_projected = input_projected.permute(0, 1, 3, 2)  # [B, N, GRU_n_dim, T]
        probs = self.graph_learner(x)  # [B, head, N, N]
        mask_loc = torch.eye(self.num_nodes, dtype=bool).to(cfg.device)
        probs_reshaped = probs.masked_select(~mask_loc).view(B, self.head, self.num_nodes * (self.num_nodes - 1)).to(cfg.device)
        probs_reshaped = probs_reshaped.permute(0, 2, 1)
        prob = F.softmax(probs_reshaped, -1) # [B, N(N-1), type]
        edges = gumbel_softmax(torch.log(prob + 1e-5), tau=self.temperature, hard=True).to(cfg.device)

        adj_list = torch.ones(self.head, B, self.num_nodes, self.num_nodes).to(cfg.device)
        mask = ~torch.eye(self.num_nodes, dtype=bool).unsqueeze(0).unsqueeze(0).to(cfg.device)
        mask = mask.repeat(self.head, B, 1, 1).to(cfg.device)
        adj_list[mask] = edges.permute(2, 0, 1).flatten() # List[B, N, N]
        state_for_output = torch.zeros(input_projected.shape).to(cfg.device)
        state_for_output = (state_for_output.unsqueeze(0)).repeat(self.head - 1, 1, 1, 1, 1)

        for head in range(self.head - 1):

            state_for_output[head, ...] = self.encoder(input_projected, adj_list[head + 1, ...], head)

        state_for_output2 = torch.mean(state_for_output, 0).permute(0, 1, 3, 2)
        output = self.linear_out(state_for_output2).squeeze(-1)[..., -1 - self.output_seq_len:-1]
        output = output.view(-1, self.output_seq_len) # (b * n_nodes, output_seq_len)
        
        return prob, output


class GRELENTrainer(ReconstructionTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # type: ignore
    
    def compute_loss(self, batch, split):
        batch = batch.to(cfg.device)
        true = batch.x
        adj, edge_index, edge_attr = None, None, None

        prob, recon = self.model(batch, return_graph=True)
        adj_mat = self.model.reshape_edges(prob) # [type, b, n_nodes, n_nodes]
        adj = adj_mat[0].detach() # [b, n_nodes, n_nodes] grelen uses the first graph for AD
        recon = recon.view(-1, cfg.dataset.n_nodes, cfg.dataset.window_size-1)
        true = true.view(-1, cfg.dataset.n_nodes, cfg.dataset.window_size)[:, :, 1:]
        
        loss_kl = kl_categorical(torch.mean(prob, 1), self.model.log_prior, 1)
        loss_nll = nll_gaussian(recon, true)
        loss = loss_kl + loss_nll
        self.custom_stats[split]['kl_loss'] = loss_kl.item()
        self.custom_stats[split]['nll_loss'] = loss_nll.item()
    
        recon = recon.view(-1, cfg.dataset.window_size-1)
        true = true.view(-1, cfg.dataset.window_size-1)
        
        if self.task.track_graph:
            return loss, recon, true, (edge_index, edge_attr, adj)
        return loss, recon, true


def get_grelen_ad_score(
    adj_probs: np.ndarray, 
    ):
    """
    probs: [b, n_nodes, n_nodes]
    
    Anomaly score based on graph relation inference (GreLeN).
    Code adapted from https://github.com/Vicky-51/GRELEN/blob/main/test_grelen.py
    Point Adjustment(PA) removed because of unfair comparison justified by 
    Towards a Rigorous Evaluation of Time-series Anomaly Detection https://arxiv.org/abs/2109.05257 
    """
    n_nodes = adj_probs.shape[1]
    w = 2 # moving window size, fixed to two https://github.com/Vicky-51/GRELEN/blob/main/config_files/SWAT_config.py#L36
    
    def moving_average(x, w):
        """Moving average of x with window size w"""
        return np.convolve(x, np.ones(w), 'valid') / w

    total_out_degree_move_filtered = np.zeros((adj_probs.shape[0] - w + 1, n_nodes))
    for i in range(n_nodes):
        y = (np.mean(adj_probs[:, :, i], -1))
        xx = moving_average(y, w)
        total_out_degree_move_filtered[:, i] = abs(y[w-1:] - xx) # added abs here that was originally missing

    score = np.mean(total_out_degree_move_filtered, 1)
    return score
