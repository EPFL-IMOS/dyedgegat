
import numbers
import sys
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import torch
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from .metric.loss import (
    masked_mae_loss, 
    masked_mape_loss,
    masked_mse_loss,
    masked_rmse_loss,
)

from .config import cfg
from .utils.io import dict_to_json, dict_to_neptune, dict_to_tb, makedirs
from .utils.device import get_current_gpu_usage
from .utils.plots import plot_node_forecasting, visualize_attention


def setup_printing(level=logging.INFO):
    cur_time = datetime.now()
    id = cur_time.strftime("%Y%m%d_%H%M%S")
    logging.root.handlers = []
    logging_cfg = {'level': level, 'format': f'%(levelname)-8s %(message)s'}
    h_stdout = logging.StreamHandler(sys.stdout)
    if cfg.print == 'file':
        h_file = logging.FileHandler(f'{cfg.run_dir}/{id}.log')
        logging_cfg['handlers'] = [h_file]
    elif cfg.print == 'stdout':
        logging_cfg['handlers'] = [h_stdout]
    elif cfg.print == 'both':
        h_file = logging.FileHandler(f'{cfg.run_dir}/{id}.log')
        logging_cfg['handlers'] = [h_file, h_stdout]
    else:
        raise ValueError('Print option not supported')
    logging.basicConfig(**logging_cfg)
    logging.info(cur_time.strftime('%c'))


class Logger(object):
    def __init__(
        self, 
        name='train', 
        neptune_writer=None,
        scaler=None
    ):
        self.name = name
        self.scaler = scaler
        self.neptune_writer = neptune_writer

        self._epoch_total = cfg.optim.max_epochs
        self._time_total = 0  # won't be reset
        self._iter_total = 0  # won't be reset

        self.out_dir = '{}/{}'.format(cfg.run_dir, name)
        makedirs(self.out_dir)

        if cfg.tensorboard_each_run:
            self.tb_writer = SummaryWriter(self.out_dir)

        self.reset()
    
    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def reset(self):
        self._iter = 0
        self._size_current = 0
        self._loss = 0
        self._lr = 0
        self._true = []
        self._pred = []
        self._time_used = 0
        self._custom_stats = {}

    # basic properties
    def basic(self):
        stats = {
            'loss': round(self._loss / self._size_current, cfg.round),
            'lr': round(self._lr, cfg.round),
            'params': self._params,
            'time_iter': round(self.time_iter(),  cfg.round),
        }
        gpu_memory = get_current_gpu_usage()
        if gpu_memory > 0:
            stats['gpu_memory'] = gpu_memory
        return stats

    def update_stats(self, target, pred, loss, lr, time_used, params, **kwargs):
        self._iter += 1
        self._iter_total += 1
        self._true.append(target)
        self._pred.append(pred)
        batch_size = int(target.shape[0] / cfg.dataset.n_nodes)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used
        for key, val in kwargs.items():
            if val is not None:
                if key not in self._custom_stats:
                    if isinstance(val, numbers.Number):
                        self._custom_stats[key] = val * batch_size 
                    else:
                        self._custom_stats[key] = [val]
                else:
                    if isinstance(val, numbers.Number):
                        self._custom_stats[key] += val * batch_size
                    else:
                        self._custom_stats[key] += [val]


        if cfg.tensorboard_iter:
            iter_stats = {
                'loss_iter': loss,
                'rmse_iter': masked_rmse_loss(target.view(pred.shape), pred, squared=False)
            }
            dict_to_tb(iter_stats, self.tb_writer, self._iter_total)

    def update_custom_stats(self, **kwargs):
        for key, val in kwargs.items():
            if val is not None and key not in self._custom_stats and isinstance(val, numbers.Number):
                self._custom_stats[key] = val

    def time_iter(self):
        return self._time_used / self._iter

    def eta(self, epoch_current):
        epoch_current += 1  # since counter starts from 0
        time_per_epoch = self._time_total / epoch_current
        return time_per_epoch * (self._epoch_total - epoch_current)
    
    # customized input properties
    def custom(self):
        if len(self._custom_stats) == 0:
            return {}
        out = {}
        for key, val in self._custom_stats.items():
            if isinstance(val, numbers.Number):
                if 'loss' in key:
                    val = val / self._size_current
                out[key] = round(float(val), cfg.round)
            elif isinstance(val, int):
                out[key] = val
            elif isinstance(val, list):
                pass
        return out
    
    def regression(self):
        true, pred = torch.cat(self._true), torch.cat(self._pred)
        true = true.view(-1).nan_to_num()
        pred = pred.view(-1).nan_to_num()
        return {
            'mae': round(float(mean_absolute_error(true, pred)), cfg.round),
            'mape': round(float(mean_absolute_percentage_error(true, pred)), cfg.round),
            'mse': round(float(mean_squared_error(true, pred)), cfg.round),
            'rmse': round(float(np.sqrt(mean_squared_error(true, pred))), cfg.round)
    }
    
    def regression_traffic(self):
        true, pred = torch.cat(self._true), torch.cat(self._pred)
        true = true.view(-1).nan_to_num()
        pred = pred.view(-1).nan_to_num()
        return {
            'mae': round(float(masked_mae_loss(true, pred)), cfg.round),
            'mape': round(float(masked_mape_loss(true, pred)), cfg.round),
            'mse': round(float(masked_mse_loss(true, pred)), cfg.round),
            'rmse': round(float(masked_rmse_loss(true, pred)), cfg.round)
        }
    
    def draw_attentions(self, adj, cur_epoch, num=3):
        adj = np.vstack(adj)
        delta = adj.shape[0] // num
        for i in range(num):
            f = visualize_attention(adj[i*delta])
            self.tb_writer.add_figure(f'figures/learned_graph/{i}', f, global_step=cur_epoch)

    def draw_weights(self, model, cur_epoch):  
        if cfg.model.dyedgegat.infer_graph:
            self.tb_writer.add_histogram('histograms/feat_edge_att', model.feat_edge_layer.att.detach().flatten(), global_step=cur_epoch, bins='tensorflow')
        if cfg.model.dyedgegat.infer_temporal_edge:
            self.tb_writer.add_histogram('histograms/temp_edge_att', model.temp_edge_layer.att.detach().flatten(), global_step=cur_epoch, bins='tensorflow')
        if cfg.model.dyedgegat.gnn_type=='gcn':
            for i in range(cfg.model.dyedgegat.num_gnn_layers):
                if 'dygat' in cfg.model.type:
                    weight = model.gnn_layers[i].lin.weight
                else:
                    weight = model.gnn_layers.convs[i].lin.weight
                self.tb_writer.add_histogram(f'histograms/gcn_{i}', weight.detach().flatten(), global_step=cur_epoch, bins='tensorflow')
        if cfg.model.dyedgegat.gnn_type=='gat':
            for i in range(cfg.model.dyedgegat.num_gnn_layers):
                if 'dygat' in cfg.model.type:
                    att = model.gnn_layers[i].att
                else:
                    att = model.gnn_layers.convs[i].att
                self.tb_writer.add_histogram(f'histograms/gat_{i}', att.detach().flatten(), global_step=cur_epoch, bins='tensorflow')
                    
        # self.tb_writer.add_histogram(f'histograms/gru_{i}', model.gnn_layers[i].lin.detach().flatten(), global_step=cur_epoch, bins='tensorflow')

    def log_figure(self, fig, figure_name, cur_epoch):
        if cfg.tensorboard_each_run:
            self.tb_writer.add_figure(figure_name, fig, global_step=cur_epoch)
        if cfg.neptune_each_run:
            self.neptune_writer[figure_name].upload(fig)

    def draw_node_pred(self, cur_epoch):
        true, pred = torch.cat(self._true).cpu().numpy(), torch.cat(self._pred).cpu().numpy()
        if cfg.tensorboard_each_run:
            f = plot_node_forecasting(pred[:, 0], true[:, 0])
            self.tb_writer.add_figure('figures/nodes_step_first', f, global_step=cur_epoch)
            f = plot_node_forecasting(pred[:, -1], true[:, -1])
            self.tb_writer.add_figure('figures/nodes_step_last', f, global_step=cur_epoch)

    def write_best_custom_to_neptune(self, best_epoch):
        # neptune
        if self.neptune_writer is not None:
            custom_stats = self.custom()
            dict_to_neptune('best', custom_stats, self.neptune_writer, best_epoch)

    def write_epoch(self, cur_epoch, is_best=False):
        basic_stats = self.basic()
        epoch_stats = {'epoch': cur_epoch}
        eta_stats = {'eta': round(self.eta(cur_epoch), cfg.round)}
        regression_stats = self.regression()
        custom_stats = self.custom()

        if self.name == 'train':
            stats = {**epoch_stats, **basic_stats, **regression_stats, **eta_stats, **custom_stats}
        else:
            stats = {**epoch_stats, **basic_stats, **regression_stats, **custom_stats}
        logging.info(f'[{self.name.upper()}]\t: {stats}')

        # json
        dict_to_json(stats, f'{self.out_dir}/stats.json')
        # tensorboard
        if cfg.tensorboard_each_run:
            dict_to_tb(stats, self.tb_writer, cur_epoch)
        # neptune
        if self.neptune_writer is not None:
            dict_to_neptune(self.name, stats, self.neptune_writer, cur_epoch)
            if is_best and self.name == 'test':
                dict_to_neptune('best', stats, self.neptune_writer, cur_epoch)
        self.reset()

    def close(self):
        if cfg.tensorboard_each_run:
            self.tb_writer.close()
        if self.neptune_writer is not None:
            self.neptune_writer.stop()