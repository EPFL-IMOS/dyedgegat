# -*- coding: utf-8 -*-

"""
@Author: mengjiezhao
@Date: 06.05.22
@Credit: ...
"""
import logging
import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, min_epochs=1000, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, cur_epoch):

        if val_loss >  self.val_loss_min - self.delta:
            self.counter += 1
            logging.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and cur_epoch > self.min_epochs:
                self.early_stop = True
        else:
            self.val_loss_min = val_loss
            self.counter = 0
