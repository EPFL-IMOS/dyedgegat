import torch
import torch.nn as nn

from ...utils.init import init_weights
from ..dict import ACTIVATION_DICT, NORM_LAYER_DICT


class Autoencoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dims, 
        activation='relu', 
        dropout_prob=0.5,
        norm_func=None,
        **kwargs
    ):
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        
        # Encoder layers
        enc_layers = []
        dims = [input_dim] + hidden_dims
        
        act = ACTIVATION_DICT[activation]
        norm_func = NORM_LAYER_DICT[norm_func] if norm_func is not None else None
        
        for i in range(len(hidden_dims)):
            enc_layers.append(nn.Linear(dims[i], dims[i+1]))
            # add norm, act, dropout of not the first or last layer
            if (i != 0 and i != len(hidden_dims) - 1) and norm_func is not None:
                enc_layers.append(norm_func(dims[i+1]))
            enc_layers.append(act())
            if (i != 0 and i != len(hidden_dims) - 1) and dropout_prob > 0.:
                enc_layers.append(nn.Dropout(p=dropout_prob))
        
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder layers
        dec_layers = []
        dims = dims[::-1]
        
        for i in range(len(hidden_dims)):
            dec_layers.append(nn.Linear(dims[i], dims[i+1]))
            # add norm, dropout of not not the first or last layer
            if i!= 0 and i != len(hidden_dims) - 1 and norm_func is not None:
                dec_layers.append(norm_func(dims[i+1]))
            if i != len(hidden_dims) - 1:
                dec_layers.append(act())
            if i!= 0 and i != len(hidden_dims) - 1 and dropout_prob > 0.:
                dec_layers.append(nn.Dropout(p=dropout_prob))

        self.decoder = nn.Sequential(*dec_layers)
        self.apply(init_weights)
    
    def forward(self, batch, **kwargs):
        if not isinstance(batch, torch.Tensor):
            x = batch.x  # (b * n_nodes, 1)
            original_shape = x.shape
            # Change input shape from (b * n_nodes, 1) to (b, n_nodes)
            x_input = x.view(-1, self.input_dim)  # (b, n_nodes)
        else:
            x_input = batch
            original_shape = batch.shape
            
        z = self.encoder(x_input)
        x_recon = self.decoder(z)
        x_recon = x_recon.view(original_shape)
        return x_recon


if __name__ == '__main__':
    from torchinfo import summary
        
    model = Autoencoder(input_dim=10, hidden_dims=[16, 8, 2], dropout_prob=0.1, norm_func='batch')
    print(model)

    summary(model, (128, 10))