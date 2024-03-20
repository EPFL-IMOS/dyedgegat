from ..config import config


def is_eval_epoch(cur_epoch, eval_period, max_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
            (cur_epoch + 1) % eval_period == 0 or
            cur_epoch == 0 or
            (cur_epoch + 1) == max_epoch
    )


def is_checkpoint_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
            (cur_epoch + 1) % config['train']['checkpoint_period'] == 0 or
            (cur_epoch + 1) == config['train']['max_epoch']
    )

