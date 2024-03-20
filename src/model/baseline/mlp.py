import torch
import torch.nn as nn

from ..dict import ACTIVATION_DICT, NORM_LAYER_DICT
from ...utils.init import init_weights


class MLP(nn.Module):
    def __init__(self, 
                 input_dim, hidden_dims, output_dim, activation='relu', 
                 dropout_prob=0.5, do_norm=True, 
                 norm_func='batch',
                 drop_first_layer=True,
                 activation_after_last_layer=False,
                 ):
        super(MLP, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        act = ACTIVATION_DICT[activation]
        norm = NORM_LAYER_DICT[norm_func] if do_norm else None
        
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if do_norm:
                layers.append(norm(dims[i+1]))
            layers.append(act())
            if not (i == 0 and not drop_first_layer):
                layers.append(nn.Dropout(p=dropout_prob))
        
        layers.append(nn.Linear(dims[-1], output_dim))
        if activation_after_last_layer:
            layers.append(act())
        self.layers = nn.Sequential(*layers)
        self.apply(init_weights)
    
    def forward(self, batch, **kwargs):
        if not isinstance(batch, torch.Tensor):
            x = batch.x  # (b * n_nodes, 1)
            # Change input shape from (b * n_nodes, 1) to (b, n_nodes)
            x = x.view(-1, self.input_dim)  # (b, n_nodes)
        else:
            x = batch
        out = self.layers(x)
        return out


if __name__ == '__main__':
  from torchinfo import summary
      
  model = MLP(input_dim=4, output_dim=14, hidden_dims=[32, 64, 32])
  print(model)

  summary(model, (128, 4))