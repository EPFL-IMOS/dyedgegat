"""
https://github.com/chaoshangcs/GTS/blob/main/model/pytorch/loss.py
"""
import torch
import torch.nn.functional as F


def mae_loss(y_pred, y_true):
    loss = torch.abs(y_pred - y_true)
    return loss.mean()


def rmse_loss(y_pred, y_true):
    return torch.sqrt(F.mse_loss(y_pred, y_true))


def mse_loss(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mape_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(torch.div(y_true - y_pred, y_true))
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def masked_rmse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())


def masked_mse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def kl_categorical(y_pred, log_prior, num_atoms, eps=1e-16):
    """ Custom implementation of the Kullback-Leibler (KL) divergence for categorical distributions
    https://github.com/Vicky-51/GRELEN/blob/main/lib/utils.py#L87
    """
    kl_div = y_pred * (torch.log(y_pred + eps) - log_prior)
    return kl_div.sum() / (num_atoms * y_pred.size(0))


def nll_gaussian(y_pred, target, variance=1, add_const=False):
    """ Calculates the negative log-likelihood (NLL) under a Gaussian distribution
    https://github.com/Vicky-51/GRELEN/blob/main/lib/utils.py#L92
    """
    neg_log_p = ((y_pred - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))
