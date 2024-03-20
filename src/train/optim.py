import torch

from ..config import cfg
from ..metric.loss import (
    masked_mae_loss, 
    masked_mse_loss, 
    masked_rmse_loss,
    mae_loss, 
    mse_loss, 
    rmse_loss,
    
)

def create_criterion(crit_type='rmse', mask_loss=False):
    if crit_type == 'rmse':
        criterion = masked_rmse_loss if mask_loss else rmse_loss
    elif crit_type == 'mse':
        criterion = masked_mse_loss if mask_loss else mse_loss
    elif crit_type == 'mae' or crit_type == 'l1':
        criterion = masked_mae_loss if mask_loss else mae_loss
    else:
        raise ValueError(f'Scheduler {crit_type} not supported')
    return criterion        


def create_scheduler(optimizer):
    optim = cfg.optim
    
    if cfg.optim.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            gamma=optim.lr_decay,
            step_size=optim.max_epochs + 1)
    elif cfg.optim.scheduler  == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            gamma=optim.gamma,
            step_size=optim.step_size)
    elif cfg.optim.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=optim.factor,
            patience=optim.patience,
            min_lr=optim.min_lr,
        )
    else:
        raise ValueError(f'Scheduler {cfg.optim.scheduler} not supported')
    return scheduler


def create_optimizer(params):
    optim = cfg.optim
    
    params = filter(lambda p: p.requires_grad, params)
    if cfg.optim.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=optim.base_lr,
            weight_decay=cfg.optim.weight_decay)
    elif cfg.optim.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, 
            lr=optim.optimizer.base_lr,
            momentum=optim.optimizer.momentum,
            weight_decay=optim.optimizer.weight_decay)
    else:
        raise ValueError(f'Optimizer {cfg.optim.optimizer} not supported')
    return optimizer
