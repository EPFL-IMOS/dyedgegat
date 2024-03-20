import os
import logging
from pprint import pp
from torchinfo import summary

from src.cmd_args import parse_args
from src.dataset.dataset import (
    create_data_loaders, 
    load_data
)
from src.agg_runs import agg_dict_list
from src.logger import setup_printing
from src.config import load_cfg, set_out_dir, set_run_dir
from src.model import GRAPH_MODELS
from src.model.factory import create_model
from src.train.create_trainer import create_trainer
from src.train.task import create_task
from src.train.optim import create_criterion, create_optimizer, create_scheduler
from src.utils.seed import is_seed, seed_everything
from src.utils.device import auto_select_device
from src.utils.neptune import create_neptune_run_writer, get_run_id, write_test_results_to_neptune, write_cfg_to_neptune
from src.utils.io import replace_nan_by_none


def eval_seed(args):
    cfg_file, out_dir, seed, test_file = args
    
    cfg = load_cfg(cfg_file)
    cfg.out_dir = out_dir
    cfg.dataset.test_file = test_file
    cfg.seed = seed
    cfg.neptune_agg = True
    cfg.neptune_each_run = False
    cfg.task.anomaly_score_sw = 1
    cfg.task.anomaly_score_thresh_mode = 'val'
    cfg.task.anomaly_score_thresh = 0.95
    
    set_run_dir(cfg.out_dir, seed)
    
    cfg.print = 'stdout'
    setup_printing(logging.INFO)
    
    seed_everything(seed)
    auto_select_device()
    logging.info(cfg)
    
    graph_datasets, scaler = load_data(
        cfg.dataset.dir, 
        win_size=cfg.dataset.window_size,
        horizon=cfg.dataset.horizon,
        aug_ocvar_on_node=cfg.dataset.aug_ocvar_on_node,
        map_ocvar=cfg.task.train_type == 'mapping',
        return_scaler=True,
    )
    loaders = create_data_loaders(graph_datasets, batch_size=cfg.train.batch_size)
    
    model = create_model(cfg.model.type)
    logging.info(summary(model))

    optimizer = create_optimizer(model.parameters())
    scheduler = create_scheduler(optimizer)
    
    crit_type = cfg.optim.criterion
    criterion = create_criterion(crit_type, mask_loss=False)

    task = create_task(
        cfg.task.type, 
        task_train_type=cfg.task.train_type, 
        track_graph=cfg.task.track_graph and cfg.model.type in GRAPH_MODELS,
    )
    
    trainer = create_trainer(
        model, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        criterion=criterion, 
        scaler=scaler,
        task=task,
    )
    
    out = trainer.evaluate(loaders)
    return out


def run_single(args):    
    print(f"Evaluating {args.cfg_file} ...")
    
    cfg = load_cfg(args.cfg_file)
    set_out_dir(cfg.out_dir, args.cfg_file)
    if 'pronto' in cfg.dataset.dir:
        test_files = ['air_blockage.csv', 'air_leakage.csv', 'diverted_flow.csv', 'slugging.csv']
    elif 'toy' in cfg.dataset.dir:
        test_files = [f'test_cr{f}.csv' for f in [0.95, 0.9, 0.75, 0.5, 1.05, 1.1, 1.25, 1.5, 2.0]]
    else:
        raise NotImplementedError(f"Unknown dataset {cfg.dataset.dir}")
    
    for test_file in test_files:
        results = []
        for seed in os.listdir(cfg.out_dir):
            if is_seed(seed):
                out = eval_seed((args.cfg_file, cfg.out_dir, int(seed), test_file))
                results.append(out)
        
        n_runs = len(results)
        results = agg_dict_list(results)
        results['n_runs'] = n_runs
        pp(results)
        results = replace_nan_by_none(results)
        
        if cfg.neptune_agg:
            run_id = get_run_id()
            neptune_writer = create_neptune_run_writer(run_id)
            write_cfg_to_neptune(cfg, neptune_writer)
            write_test_results_to_neptune(results, neptune_writer)
            neptune_writer.stop()


if __name__ == "__main__":
    args = parse_args()
    run_single(args)
