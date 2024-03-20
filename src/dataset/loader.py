from ..config import cfg
from .dataset import (
    load_data as load_data_anomaly,
    create_data_loaders as create_data_loaders_anomaly, 
)


def load_data_and_create_dataloader():

    graph_datasets, scaler = load_data_anomaly(
        cfg.dataset.dir, 
        win_size=cfg.dataset.window_size,
        horizon=cfg.dataset.horizon,
        aug_ocvar_on_node=cfg.dataset.aug_ocvar_on_node,
        map_ocvar=cfg.task.train_type == 'mapping',
        return_scaler=True,
    )

    loaders = create_data_loaders_anomaly(graph_datasets, batch_size=cfg.train.batch_size)
    
    return loaders, scaler