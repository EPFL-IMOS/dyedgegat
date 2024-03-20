import os
import time
import torch
import logging
import numpy as np
import torch.nn as nn
from torch_geometric.utils import to_dense_adj


from .earlystopping import EarlyStopping
from ..config import cfg
from ..logger import Logger
from ..utils.checkpoint import (
    clean_ckpt, 
    is_ckpt_epoch, 
    is_draw_epoch, 
    load_ckpt, 
    load_pretrained, 
    save_ckpt
)

class BaseTrainer:
    def __init__(
        self,
        model,
        task,
        optimizer=None,
        scheduler=None,
        n_epochs=20,
        scaler=None,
        criterion = nn.MSELoss(),
        model_name='model',  
        neptune_writer=None,      
        task_train_type='reconst',
        clip_grad=True,
        max_grad_norm=1
    ):
        self.model = model
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.model_name = model_name
        self.criterion = criterion
        self.clip_grad = clip_grad
        self.max_grad_norm = max_grad_norm
        self.scaler = scaler
        self.task_train_type = task_train_type
        self.model_type = model.__class__.__name__.lower()
        self.neptune_writer = neptune_writer
        self.loggers = {split:  Logger(split, scaler=scaler, neptune_writer=self.neptune_writer) 
                        for split in ['train', 'eval', 'test']}
        self.losses = {split: [] for split in ['train', 'eval', 'test']}
        self.errors = {split: [] for split in ['train', 'eval', 'test']}
        self.custom_stats = {split: {} for split in ['train', 'eval', 'test']}
    
    def compute_loss(self, batch, split):
        raise NotImplementedError
    
    def update_loss(self, split):
        self.losses[split].append(self.loggers[split].basic()['loss'])
        self.errors[split].append(self.loggers[split].regression()[cfg.metric_best])
        
    def train_epoch(self, loader, epoch=0):
        self.model.train()
        time_start = time.time()
        for batch in loader:
            self.optimizer.zero_grad()

            out = self.compute_loss(batch, split='train')
            loss, pred, target = out[0], out[1], out[2]
            
            if torch.isnan(loss):
                logging.error("Discarding run with nan value")
                raise ValueError("Loss is nan, discarding this run!")
            
            loss.backward()
            self.optimizer.step()
            
            if torch.isnan(loss):
                raise RuntimeError("Loss is nan! Run discarded for {cfg.out_dir}")
            
            self.loggers['train'].update_stats(
                target=target.detach().cpu(),
                pred=pred.detach().cpu(),
                loss=loss.item(),
                lr=self.optimizer.param_groups[0]['lr'],
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
            
            out = self.compute_loss(batch, split)
            loss, pred, target = out[0], out[1], out[2]
        
            args = dict(
                target=target.detach().cpu(),
                pred=pred.detach().cpu(),
                loss=loss.item(),
                lr=0,
                time_used=time.time() - time_start,
                params=cfg.params,
                **self.custom_stats[split]
            )
            
            if self.task.track_graph:
                edge_index, edge_attr, adj = out[-1]
                args['adj'] = adj.cpu().numpy()
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
            if self.task.track_graph and cfg.draw_learned_graph:
                adj = self.loggers[split]._custom_stats['adj']
                self.loggers[split].draw_attentions(adj, epoch)
            if self.task.task_level == 'graph':
                self.loggers[split].draw_graph_pred(epoch)
            elif self.task.task_level == 'node':
                self.loggers[split].draw_node_pred(epoch)
        
        if split == 'eval':
            # TODO adapt to USAD
            if cfg.optim.scheduler == 'plateau':
                self.scheduler.step(self.loggers['eval'].basic()['loss'])
            else:
                self.scheduler.step()
        
    def fit(self, loaders):
        logging.info(f"Training for {self.n_epochs} epochs...")
        train_loader, val_loader, test_loader = loaders
        train_start = time.time()
        
        early_stopping = EarlyStopping(
            patience=cfg.train.early_stop_patience,
            min_epochs=cfg.train.early_stop_min,
            verbose=True
        )
        
        start_epoch = 0
        self.current_epoch = 0
        self.min_var_err = np.inf
        self.best_test_err = np.inf
        self.best_epoch = 0
        
        self.load(self.model)
        
        # load checkpoint
        if cfg.train.auto_resume:
            start_epoch = load_ckpt(self.model, self.optimizer, self.scheduler, epoch=cfg.train.epoch_resume)
            logging.info(f"Checkpoint found, starting from epoch {start_epoch}")
        if start_epoch == cfg.optim.max_epochs:
            logging.info('Checkpoint found, Task already done')
        else:
            if cfg.train.load_pretrained:
                load_pretrained(self.model, cfg.train.pretrained_model_path)
                logging.info(f'Pretrained model loaded from {cfg.train.pretrained_model_path}')
                
            logging.info(f'Start training from epoch {start_epoch}')
        
        for epoch in range(start_epoch, self.n_epochs):
            self.current_epoch = epoch
            self.train_epoch(train_loader, epoch)
            self.loggers['train'].write_epoch(epoch)
            for split, loader in zip(['eval', 'test'], [val_loader, test_loader]):
                self.eval_epoch(loader, split, epoch)
            
            # save model if eval error is lower than previous best
            if val_loader is not None:
                if self.errors['eval'][-1] < self.min_var_err:
                    eval_err = self.errors['eval'][-1]
                    test_err = self.errors['test'][-1]
                    self.min_var_err = eval_err
                    self.best_test_err = test_err
                    self.best_epoch = epoch
                    if cfg.neptune_each_run:
                        self.loggers['test'].write_best_custom_to_neptune(self.best_epoch)
                    logging.info(f"Current best epoch is {epoch} with eval err {eval_err:.5f} and test err {test_err:.5f}, saving model...")
                    self.save(self.model)
            
            self.task.eval_task_epoch(self.loggers, cur_epoch=epoch, is_best=self.best_epoch == epoch)
            
            for split, loader in zip(['eval', 'test'], [val_loader, test_loader]):
                if loader is not None:
                    self.loggers[split].write_epoch(epoch)
            
            # early stopping
            early_stopping(self.errors['eval'][-1], epoch)
            
            if early_stopping.early_stop:
                logging.info("Early stopping! Saving checkpoint...")
                save_ckpt(self.model, self.optimizer, self.scheduler, epoch)
                break
            
            # save checkpoint
            if is_ckpt_epoch(epoch):
                save_ckpt(self.model, self.optimizer, self.scheduler, epoch)
        
        if self.neptune_writer is not None:
            self.neptune_writer['best/epoch'] = self.best_epoch
            self.neptune_writer['best/val_err'] = self.min_var_err
            self.neptune_writer['best/test_err'] = self.best_test_err

        self.evaluate(loaders)

        for logger in self.loggers.values():
            logger.close()
        
        train_time = int(time.time() - train_start)
                
        if cfg.train.ckpt_clean:
            clean_ckpt()
        
        logging.info(f'Task done in {train_time}s, results saved in {cfg.run_dir}')
    
    def evaluate(self, loaders, return_raw=False):
        self.load(self.model)
        logging.info(f"Evaluating best model from epoch {self.best_epoch}...")
        val_loader, test_loader = loaders[1], loaders[2]
        for split, loader in zip(['eval', 'test'], [val_loader, test_loader]):
            self.eval_epoch(loader, split)
        out = self.task.eval_task_epoch(self.loggers, 
                                        cur_epoch=self.best_epoch, is_best=True,
                                        return_raw=return_raw)
        self.loggers['test'].write_best_custom_to_neptune(self.best_epoch)
        self.loggers['test'].write_epoch(self.best_epoch)
        return out
    
    def save(self, model, model_path=None):
        if model_path is None:
            model_path = f"{cfg.run_dir}/{self.model_name}.pt"
        torch.save(
            {
                'model_state': model.state_dict(),
                'min_var_err': self.min_var_err,
                'best_test_err': self.best_test_err,
                'best_epoch': self.best_epoch,
             }, model_path)
        
    def load(self, model, model_path=None):
        """
        loads the prediction model's parameters
        """
        if model_path is None:
            model_path = f"{cfg.run_dir}/{self.model_name}.pt"
        if os.path.exists(model_path):
            model_ckpt = torch.load(model_path, map_location=cfg.device)
            model.load_state_dict(model_ckpt['model_state'])
            self.min_var_err = model_ckpt.get('min_var_err', np.nan)
            self.best_test_err = model_ckpt.get('best_test_err', np.nan)
            self.best_epoch = model_ckpt.get('best_epoch', 0)
            self.current_epoch = self.best_epoch # init current epoch
            logging.info(f"Model {model.__class__.__name__} saved in {model_path} loaded to {cfg.device}, " + 
                          f"best epoch {self.best_epoch} with eval err {self.min_var_err:.5f} and test err {self.best_test_err:.5f}")        
        else:
            logging.info(f"No model found at {model_path}, starting from scratch")


class MappingTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
         
    def compute_loss(self, batch, split):
        batch.to(torch.device(cfg.device))
        y_pred = self.model(batch)
        b = y_pred.shape[0] # batch size
        y_true = batch.y
        y_true = y_true.view(b, -1)
        loss = self.criterion(y_pred, y_true)
        return loss, y_pred.unsqueeze(2), y_true.unsqueeze(2)


class ReconstructionTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # type: ignore
    
    def compute_loss(self, batch, split):
        batch = batch.to(cfg.device)
        true = batch.x
        adj, edge_index, edge_attr = None, None, None

        if self.task.track_graph:
            recon, edge_index, edge_attr = self.model(batch, return_graph=self.task.track_graph)
            edge_index = edge_index.detach()
            if edge_attr is not None:
                edge_attr = edge_attr.detach()
            adj = to_dense_adj(edge_index, edge_attr=edge_attr, batch=batch.batch)
            if adj.ndim == 4:
                adj = adj.squeeze(3)
            recon = recon.view(true.shape)
            loss = self.criterion(recon, true)

        else:
            recon = self.model(batch)
            recon = recon.view(true.shape)
            adj = None
            loss = self.criterion(recon, true)

        if self.task.track_graph:
            return loss, recon, true, (edge_index, edge_attr, adj)
        return loss, recon, true


class ForecastingTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, batch, split):
        batch = batch.to(cfg.device)
        true = batch.y
        if self.task.track_graph:
            pred, edge_index, edge_attr = self.model(batch, return_graph=self.task.track_graph)
            edge_index = edge_index.detach()
            if edge_attr is not None:
                edge_attr = edge_attr.detach()
            adj = to_dense_adj(edge_index, edge_attr=edge_attr, batch=batch.batch)
            if adj.ndim == 4:
                adj = adj.squeeze(3)
        else:
            pred = self.model(batch)
            adj = None
        
        pred = pred.view(true.shape)

        loss = self.criterion(true, pred)
                
        if self.task.track_graph:
            return loss, pred, true, (edge_index, edge_attr, adj)
        return loss, pred, true


class CombinedTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for split in ['train', 'eval', 'test']:
            self.custom_stats[split]['fc_true'] = []
            self.custom_stats[split]['fc_pred'] = []
    
    def compute_loss(self, batch, split):
        batch = batch.to(cfg.device)
        true_pred = batch.y
        true_recon = batch.x
        if self.task.track_graph:
            pred, recon, edge_index, edge_attr = self.model(batch, return_graph=self.task.track_graph)
            edge_index = edge_index.detach()
            if edge_attr is not None:
                edge_attr = edge_attr.detach()
            adj = to_dense_adj(edge_index, edge_attr=edge_attr, batch=batch.batch)
            if adj.ndim == 4:
                adj = adj.squeeze(3)
        else:
            pred, recon = self.model(batch)
            adj = None
        
        pred = pred.view(true_pred.shape)
        recon = recon.view(true_recon.shape)

        loss_pred = self.criterion(true_pred, pred)
        loss_recon = self.criterion(true_recon, recon)
        
        self.custom_stats[split]['fc_loss'] = loss_pred.item()
        self.custom_stats[split]['recon_loss'] = loss_recon.item()
        self.custom_stats[split]['fc_true'] = true_pred.detach().cpu()
        self.custom_stats[split]['fc_pred'] = pred.detach().cpu()
        loss = loss_pred + loss_recon
                
        if self.task.track_graph:
            return loss, recon, true_recon, (edge_index, edge_attr, adj)
        return loss, recon, true_recon