import torch
import torch.nn as nn

from ..dict import ACTIVATION_DICT, NORM_LAYER_DICT
from ...utils.init import init_weights


class CNN(nn.Module):
    '''
    A 1D-CNN model that is based on the paper "Fusing physics-based and deep learning models for prognostics"
    from Manuel Arias Chao et al. (with batchnorm layers)
    '''
    
    def __init__(self, 
                 input_dim=18, 
                 output_dim=None,
                 window_size=50,
                 n_channels=10, 
                 kernel_size=10, 
                 stride=1,
                 n_layers=3,
                 n_hidden=50,
                 dropout_prob=0.5, 
                 activation='relu',
                 norm_func='batch',
                 padding='same'):
        """
        Args:
            n_features (int, optional): number of input features. Defaults to 18.
            window (int, optional): sequence length. Defaults to 50.
            n_ch (int, optional): number of channels (filter size). Defaults to 10.
            n_k (int, optional): kernel size. Defaults to 10.
            n_hidden (int, optional): number of hidden neurons for regressor. Defaults to 50.
            n_layers (int, optional): _description_. Defaults to 3.
        """
        super().__init__()
        
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.window_size = window_size
        
        act = ACTIVATION_DICT[activation]
        
        cnn_layers = []
        for i in range(n_layers):
            in_channels = input_dim if i == 0 else n_channels
            out_channels = n_channels if i != n_layers - 1 else 1
            cnn_layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            cnn_layers.append(cnn_layer)
            if i!= n_layers - 1:
                if norm_func is not None:
                    cnn_layers.append(NORM_LAYER_DICT[norm_func](n_channels))
                if dropout_prob > 0.:
                    cnn_layers.append(nn.Dropout(dropout_prob))
            cnn_layers.append(act())
        
        self.cnn_layers = nn.Sequential(*cnn_layers)
        
        self.mlp = nn.Sequential(nn.Linear(window_size, n_hidden), 
                                 act(), nn.Linear(n_hidden, output_dim))
        
        self.apply(init_weights)
        
    def forward(self, batch):
        if not isinstance(batch, torch.Tensor):
            x = batch.x  # (b * n_nodes, window_size)
        
            # Change input shape from (b * n_nodes, window_size) to (b, n_nodes, window_size)
            x_input = x.view(-1, self.input_dim, x.shape[-1])  # (b, n_nodes, window_size)
        else:
            x_input = batch
            
        # Propagate input through Conv-Layers
        x = self.cnn_layers(x_input) # (b, 1, window_size)
        x = x.squeeze(1)  # (b, window_size)

        x = self.mlp(x)
        out = x.view(x.shape[0], -1)  # (b, out_dim)
        
        return out


if __name__ == '__main__':
    from torchinfo import summary
    seq_length = 50
    n_nodes = 5
    batch_size = 2
    model = CNN(
        input_dim=n_nodes, 
        window_size=seq_length,
        n_channels=10,
        n_layers=4,
        output_dim=1, 
        kernel_size=5, 
        padding='same',
        norm_func='batch',
        dropout_prob=0.3
    )
    print(model)
    summary(model, (batch_size, n_nodes, seq_length))
