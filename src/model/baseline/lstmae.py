"""
Implementation of multivariate time series anomaly detection model introduced in Malhotra, P., Ramakrishnan, A.,
        Anand, G., Vig, L., Agarwal, P. and Shroff, G., 2016. LSTM-based encoder-decoder for multi-sensor anomaly detection.

ENCDEC_AD: https://arxiv.org/pdf/1607.00148.pdf
"""

import torch
import torch.nn as nn
from torchinfo import summary
from model.dict import ACTIVATION_DICT, NORM_LAYER_DICT

from ...utils.init import init_weights


class LSTM_AE(nn.Module):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            activation='relu',
            norm_func=None,
            dropout_prob=0.,
            **kwargs
            ) -> None:
        """
        LSTM Variational AutoEncoder
        :param input_dim: 
        :param latent_dim: 
        :param kwargs:
        """

        super(LSTM_AE, self).__init__()

        self.input_dim = input_dim

        self.act = ACTIVATION_DICT[activation]()
        self.norm = NORM_LAYER_DICT[norm_func](hidden_dim) if norm_func is not None else None

        self.encoder = nn.LSTM(
            input_dim, 
            hidden_dim, 
            batch_first=True,
            dropout=dropout_prob,
        )
        
        self.decoder = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            batch_first=True,
            dropout=dropout_prob,
        )
        
        self.linear = nn.Linear(hidden_dim, input_dim)
        
        self.apply(init_weights)

    def forward(self, batch):
        if not isinstance(batch, torch.Tensor):
            x = batch.x  # (b * n_nodes, window_size)
            # Change input shape from (b * n_nodes, window_size) to (b, n_nodes, window_size)
            x = x.view(-1, self.input_dim, x.shape[1])  # (b, n_nodes, window_size)
        else:
            x = batch
        seq_len = x.shape[2]

        _, (h, c) = self.encoder(x.permute(0, 2, 1))  # (b, window_size, n_nodes)
        
        h = h[-1] # take the last layer

        # Apply activation function
        # h = self.act(h)

        # Apply normalization if specified
        if self.norm is not None:
            out = self.norm(h)

        h = h.unsqueeze(1).repeat(1, seq_len, 1) # (b, seq, hidden_dim)
        
        out, _ = self.decoder(h)
        out = torch.flip(out, [1]) # reverse the sequence (b, window_size, n_nodes) 
        
        out = self.linear(out) # (b, window_size, n_nodes) 
        
        out = out.permute(0, 2, 1)  # (b, n_nodes, window_size) 
        out = out.reshape(-1, seq_len)
        return out


if __name__ == '__main__':
    window = 15
    feature_dim = 7
    hidden_dim = 20
    batch_size = 2
    
    encdec_ad = LSTM_AE(
        input_dim=feature_dim, 
        hidden_dim=hidden_dim,
        latent_dim=2,
        num_layers=1,
    )
    print(encdec_ad)
    summary(encdec_ad, (batch_size, feature_dim, window))
