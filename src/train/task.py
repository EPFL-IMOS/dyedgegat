import logging
import numpy as np
import torch

from ..config import cfg
from ..metric.anomaly_score import get_best_prediction_scores
from ..utils.plots import (
    plot_confusion_matrix, 
    plot_roc_auc, 
)


def create_task(task_type, **task_args):
    if task_type == 'anomaly':
        task = AnomalyDetectionTask(**task_args)
    elif task_type == 'forecasting':
        task = ForecastingTask(**task_args)
    else:
        raise ValueError(f"Invalid task type: '{task_type}', should be one of 'anomaly', 'forecasting'.")
    
    logging.info(f"Create {task.__class__.__name__}.")
    return task


class BaseTask:
    def __init__(self, 
        task_train_type='reconst',
        track_graph=False,
    ):
        self.task_train_type=task_train_type
        self.track_graph=track_graph
        
    def log_task_figures(self, loggers, **kwargs):
        raise NotImplementedError
    
    def eval_task_epoch(self, **kwargs):
        raise NotImplementedError


class ForecastingTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'forecasting'
        self.task_level = 'node'


class AnomalyDetectionTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'anomaly'
        self.task_level = 'node'
        
    def eval_task_epoch(self, loggers, cur_epoch, 
                        is_best, state='train',
                        return_raw=False, **kwargs):
        val_preds = torch.cat(loggers['eval']._pred).detach().cpu().numpy()
        val_trues = torch.cat(loggers['eval']._true).detach().cpu().numpy()
        
        test_preds = torch.cat(loggers['test']._pred).detach().cpu().numpy()
        test_trues = torch.cat(loggers['test']._true).detach().cpu().numpy()
        
        window_size = cfg.dataset.window_size
        n_nodes = cfg.dataset.n_nodes
        
        if self.task_train_type == 'forecast':
            window_size = cfg.dataset.horizon
        elif self.task_train_type == 'mapping':
            window_size = 1
            n_nodes = cfg.model.output_dim

        if cfg.model.type == 'grelen':
            window_size -= 1 # grelen does not reconstruct the first time step
        
        true_labels = torch.cat(loggers['test']._custom_stats['label']).detach().cpu().numpy()
        fault_labels = torch.cat(loggers['test']._custom_stats['fault']).detach().cpu().numpy() if 'fault' in loggers['test']._custom_stats else None
        
        batch_size = int(true_labels.shape[0])

        if cfg.model.type == 'usad':
            val_reconst_1 = torch.cat(loggers['eval']._custom_stats['w1']).detach().cpu().numpy().reshape(-1, n_nodes, window_size)
            val_reconst_2 = torch.cat(loggers['eval']._custom_stats['w3']).detach().cpu().numpy().reshape(-1, n_nodes, window_size)
            test_reconst_1 = torch.cat(loggers['test']._custom_stats['w1']).detach().cpu().numpy().reshape(-1, n_nodes, window_size)
            test_reconst_2 = torch.cat(loggers['test']._custom_stats['w3']).detach().cpu().numpy().reshape(-1, n_nodes, window_size)
            val_preds = (val_reconst_1, val_reconst_2)
            test_preds = (test_reconst_1, test_reconst_2)
            val_trues = val_trues.reshape(-1, n_nodes, window_size)
            test_trues = test_trues.reshape(-1, n_nodes, window_size)

        elif cfg.task.train_type == 'combined':
            val_forecast_pred = torch.cat(loggers['eval']._custom_stats['fc_pred']).detach().cpu().numpy().reshape(-1, cfg.dataset.n_nodes, cfg.dataset.horizon)
            val_forecast_true = torch.cat(loggers['eval']._custom_stats['fc_true']).detach().cpu().numpy().reshape(-1, cfg.dataset.n_nodes, cfg.dataset.horizon)
            
            test_forecast_pred = torch.cat(loggers['test']._custom_stats['fc_pred']).detach().cpu().numpy().reshape(-1, cfg.dataset.n_nodes, cfg.dataset.horizon)
            test_forecast_true = torch.cat(loggers['test']._custom_stats['fc_true']).detach().cpu().numpy().reshape(-1, cfg.dataset.n_nodes, cfg.dataset.horizon)

            val_preds = val_preds.reshape(-1, n_nodes, window_size)
            val_trues = val_trues.reshape(-1, n_nodes, window_size)
            test_preds = test_preds.reshape(-1, n_nodes, window_size)
            test_trues = test_trues.reshape(-1, n_nodes, window_size)
            
            val_preds = (val_preds, val_forecast_pred)
            val_trues = (val_trues, val_forecast_true)
            test_preds = (test_preds, test_forecast_pred)
            test_trues = (test_trues, test_forecast_true)

        else:
            val_preds = val_preds.reshape(-1, n_nodes, window_size)
            val_trues = val_trues.reshape(-1, n_nodes, window_size)
            test_preds = test_preds.reshape(-1, n_nodes, window_size)
            test_trues = test_trues.reshape(-1, n_nodes, window_size)

        args = dict(
            anomaly_score_func=cfg.task.anomaly_score_func,
            val_preds=val_preds, 
            val_trues=val_trues, 
            test_preds=test_preds,
            test_trues=test_trues, 
            true_labels=true_labels
        )
        
        if self.track_graph or cfg.model.type == 'grelen':
            adj_val = np.vstack(loggers['eval']._custom_stats['adj'])
            adj_test = np.vstack(loggers['test']._custom_stats['adj'])
            args['adj_val'] = adj_val
            args['adj_test'] = adj_test
            
        if cfg.task.anomaly_score_func == 'node_scaled_combined':
            args['gamma'] = cfg.task.gamma
        
        out = get_best_prediction_scores(total_steps=window_size,
                                        metric=cfg.task.metric,
                                        fault_labels=fault_labels,
                                        return_scores=return_raw,
                                        **args)
        if return_raw:
            return out, args

        if return_raw:
            out = out[0]

        out['batch_size'] = batch_size        
        fpr = out.pop('fpr')
        tpr = out.pop('tpr')
        
        if is_best and state == 'train':
            # plot confusion matrix
            fig = plot_confusion_matrix(out['tn'], out['fp'], out['fn'], out['tp'], 
                                        labels=['normal', 'anomaly'], 
                                        title='Confusion matrix')

            loggers['test'].log_figure(figure_name='confusion_matrix', fig=fig, cur_epoch=cur_epoch)

            # plot auc curve
            fig = plot_roc_auc(fpr, tpr, out['auc'])
            loggers['test'].log_figure(figure_name='auc_roc', fig=fig, cur_epoch=cur_epoch)

        loggers['test'].update_custom_stats(**out)

        return out