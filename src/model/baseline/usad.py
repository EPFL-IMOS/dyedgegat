"""
USAD - UnSupervised Anomaly Detection on multivariate time series
https://github.com/manigalati/usad/blob/master/usad.py
Partially taken from 
TimeSeAD - Library for Benchmarking Multivariate Time Series Anomaly Detection
https://github.com/wagner-d/TimeSeAD/blob/master/timesead/models/reconstruction/usad.py
"""

import time
import numpy as np
import torch
import torch.nn as nn

from .mlp import MLP
from ...config import cfg
from ...utils.checkpoint import is_draw_epoch
from ...utils.init import init_weights
from ...train.optim import create_optimizer, create_scheduler
from ...train.train import ReconstructionTrainer
from ...model.dict import ACTIVATION_DICT


class USAD(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 window_size: int,
                 latent_dim: int,
                 activation='relu',
                 final_act='sigmoid',
                 norm_func=None,
                 n_hidden_layers=2,
                 dropout_prob=0.,
                 **kwargs
                 ):
        super().__init__()
        self.input_dim = input_dim
        encoder_in = input_dim * window_size
        encoder_hidden_layers = [encoder_in // 2 ** i for i in range(1, n_hidden_layers + 1)]
        self.encoder = MLP(encoder_in, encoder_hidden_layers, latent_dim, activation=activation,
                           dropout_prob=dropout_prob,
                           do_norm=norm_func is not None, norm_func=norm_func, activation_after_last_layer=True)
        decoder_hidden_layers = encoder_hidden_layers[::-1]
        final_act = ACTIVATION_DICT[final_act]()
        self.decoder1 = torch.nn.Sequential(
            MLP(latent_dim, decoder_hidden_layers, encoder_in, activation=activation,
                dropout_prob=dropout_prob,
                do_norm=norm_func is not None, norm_func=norm_func),
                final_act
        )
        self.decoder2 = torch.nn.Sequential(
            MLP(latent_dim,decoder_hidden_layers, encoder_in, 
                dropout_prob=dropout_prob, activation=activation,
                do_norm=norm_func is not None, norm_func=norm_func),
                final_act
        )
        self.apply(init_weights)

    def forward(self, batch):
        """
        :param inputs: Tuple with one tensor of shape (b, window_size, n_nodes)
        :return:
        """
        if not isinstance(batch, torch.Tensor):
            x = batch.x  # (b * n_nodes, window_size)
            # Change input shape from (b * n_nodes, window_size) to (b, n_nodes, window_size)
            x = x.view(-1, self.input_dim, x.shape[1])  # (b, n_nodes, window_size)
            x = x.permute(0, 2, 1)  # (b, window_size, n_nodes)
        else:
            x = batch
        
        x_shape = x.shape # (b, window_size, n_nodes)
        # Flatten time dimension
        x = x.reshape(x.shape[0], -1)

        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))

        # Restore time dimension
        return w1.view(x_shape), w2.view(x_shape), w3.view(x_shape)


class USADTrainer(ReconstructionTrainer):
    def __init__(self, **kwargs):
        kwargs.pop('optimizer')
        kwargs.pop('scheduler')
        super().__init__(**kwargs) # type: ignore
        
        self.warmup_epochs = cfg.model.usad.warmup_epochs
        
        self.optimizer1 = create_optimizer(list(self.model.encoder.parameters())+list(self.model.decoder1.parameters()))
        self.optimizer2 = create_optimizer(list(self.model.encoder.parameters())+list(self.model.decoder2.parameters()))
        self.scheduler1 = create_scheduler(self.optimizer1) 
        self.scheduler2 = create_scheduler(self.optimizer2)
    
    
    def compute_loss_original(self, batch, split):
        batch = batch.to(cfg.device)
        
        w1, w2, w3 = self.model(batch)
        x = batch.x.view(-1, cfg.dataset.n_nodes, cfg.dataset.window_size).permute(0, 2, 1)
        n = self.current_epoch + 1

        usad_loss1 = 1/n * torch.mean((x - w1) ** 2) + (1 - 1 / n) * torch.mean((x - w3) ** 2)
        usad_loss2 = 1/n * torch.mean((x - w2) ** 2) - (1 - 1 / n) * torch.mean((x - w3) ** 2)
        
        return usad_loss1, usad_loss2, w1, w2, w3


    def compute_loss(self, batch, split):
        batch = batch.to(cfg.device)
        
        w1, w2, w3 = self.model(batch)
        x = batch.x.view(-1, cfg.dataset.n_nodes, cfg.dataset.window_size).permute(0, 2, 1)
        n = self.current_epoch + 1
        
        if self.current_epoch < self.warmup_epochs:
            reg1, reg2 = 1, 0
        else:
            reg1 = 1 / (n - self.warmup_epochs)
            reg2 = 1 - 1 / (n - self.warmup_epochs)

        usad_loss1 = reg1 * self.criterion(x, w1) + reg2 * self.criterion(x, w3)
        usad_loss2 = reg1 * self.criterion(x, w2) - reg2 * self.criterion(x, w3)
        
        return usad_loss1, usad_loss2, w1, w2, w3
    
    def train_epoch(self, loader, epoch=0):
        self.model.train()
        time_start = time.time()
        for batch in loader:

            self.optimizer1.zero_grad()
            loss1, loss2, w1, w2, w3 = self.compute_loss(batch, split='train')
            self.custom_stats['train']['usad_loss1'] = loss1.item()
            loss1.backward()
            self.optimizer1.step()
            
            self.optimizer2.zero_grad()
            loss1, loss2, w1, w2, w3 = self.compute_loss(batch, split='train')
            self.custom_stats['train']['usad_loss2'] = loss2.item()
            loss2.backward()
            self.optimizer2.step()


            self.loggers['train'].update_stats(
                target=batch.x.detach().cpu(),
                pred=w1.detach().cpu(),
                loss=(loss1+loss2).item(),
                lr=self.optimizer1.param_groups[0]['lr'],
                time_used=time.time() - time_start,
                params=cfg.params,
                **self.custom_stats['train']
            )
            time_start = time.time()
        
        if cfg.draw_learned_graph and is_draw_epoch(epoch):
            if 'dygat' in cfg.model.type:
                self.loggers['train'].draw_weights(self.model, epoch)
            if self.task.task_level == 'graph':
                self.loggers['train'].draw_graph_pred(epoch)
        self.update_loss(split='train')
        
        # gradient clipping - this does it in place
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
    
    @torch.no_grad()
    def eval_epoch(self, loader, split='eval', epoch=0):
        self.model.eval()
        time_start = time.time()
        for batch in loader:
            
            loss1, loss2, w1, w2, w3 = self.compute_loss(batch, split='train')
            self.custom_stats[split]['usad_loss1'] = loss1.item()
            self.custom_stats[split]['usad_loss2'] = loss2.item()
            
            window_size = w1.shape[1]
            w1 = w1.permute(0, 2, 1).reshape(-1, window_size) # (b*n_nodes, window_size)
            w3 = w3.permute(0, 2, 1).reshape(-1, window_size) # (b*n_nodes, window_size)
           
            self.custom_stats[split]['w1'] = w1.detach().cpu()
            self.custom_stats[split]['w3'] = w3.detach().cpu()
        
            args = dict(
                target=batch.x.detach().cpu(),
                pred=w1.detach().cpu(),
                loss=(loss1+loss2).item(),
                lr=0,
                time_used=time.time() - time_start,
                params=cfg.params,
                **self.custom_stats[split]
            )
            
            if self.task.name == 'anomaly' and split == 'test':
                args['label'] = batch.label.detach().cpu()
                if hasattr(batch, 'fault'):
                    args['fault'] = batch.fault.detach().cpu()
            
            self.loggers[split].update_stats(
                **args
            ) # type: ignore
            time_start = time.time()
        
        self.update_loss(split=split)
        
        # visualize last batch
        if is_draw_epoch(epoch):
            self.loggers[split].draw_node_pred(epoch)
        
        if split == 'eval':
            # TODO adapt to USAD
            if cfg.optim.scheduler == 'plateau':
                self.scheduler1.step(self.loggers['eval'].custom()['usad_loss1'])
                self.scheduler2.step(self.loggers['eval'].custom()['usad_loss2'])
            else:
                self.scheduler1.step()
                self.scheduler2.step()


def get_usad_ad_score(trues, preds):
    # Input of shape (b, n_nodes, window_size), output of shape (b,)
    alpha = cfg.model.usad.alpha
    beta = 1 - alpha
    w1, w3 = preds
    # take mean along the dimension of feature of the last time step
    score_1 = np.mean((trues[:, -1, :]-w1[:, -1, :])**2, axis=-1)
    score_2 = np.mean((trues[:, -1, :]-w3[:, -1, :])**2, axis=-1)
    score = alpha * score_1 + beta * score_2
    return score