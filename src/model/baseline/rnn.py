"""
Implementation of (LSTM-AD) Long Short Term Memory Networks for Anomaly Detection in Time Series
"""

import torch
import torch.nn as nn

from src.model.dict import (
    ACTIVATION_DICT, 
    NORM_LAYER_DICT, 
    RNN_LAYER_DICT
)
from ...utils.init import init_weights

# TODO consider univariate one?


class RNN(nn.Module):
    '''
    A RNN model
    '''
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        rnn_type='gru',
        activation='relu',
        n_layers=1,
        bidirectional=False,
        norm_func=None,
        dropout_prob=0.,
    ):
        super(RNN, self).__init__()

        if output_dim is None:
            output_dim = input_dim
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.rnn = RNN_LAYER_DICT[rnn_type](
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.norm = NORM_LAYER_DICT[norm_func](hidden_dim) if norm_func is not None else None
        self.act = ACTIVATION_DICT[activation]()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply(init_weights)
        
    def forward(self, batch, **kwargs):
        if not isinstance(batch, torch.Tensor):
            x = batch.x  # (b * n_nodes, window_size)
        
            # Change input shape from (b * n_nodes, window_size) to (b, n_nodes, window_size)
            x_input = x.view(-1, self.input_dim, x.shape[1])  # (b, n_nodes, window_size)
            x_input = x_input.permute(0, 2, 1) # (b, window_size, n_nodes)
        else:
            x_input = batch
            x_input = x_input.permute(0, 2, 1) # (b, window_size, n_nodes)
        
        # RNN layer
        out, _ = self.rnn(x_input) # (b, window_size, n_nodes)
        
        # Extracting from last layer # (b, window_size, D * hidden_dim)
        out = out[:, -1, :]
        
        # Apply activation function
        out = self.act(out)
        
        # Apply normalization if specified
        if self.norm is not None:
            out = self.norm(out)
        
        # Apply dropout if specified
        out = self.dropout(out)
        
        # Fully connected layer
        out = self.fc(out)
        
        out = out.view(out.shape[0] * self.input_dim, -1)  # (b * n_nodes, out_dim)
        
        return out



if __name__ == '__main__':
    from torchinfo import summary
    seq_length = 50
    n_nodes = 10
    hidden_size = 20
    batch_size = 2
    rnn = RNN(input_dim=n_nodes, hidden_dim=hidden_size, output_dim=1, dropout_prob=0.2, rnn_type='lstm', norm_func='layer')
    summary(rnn, (batch_size, n_nodes, seq_length))

