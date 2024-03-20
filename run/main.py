import os
import torch
import logging
from torchinfo import summary

from src.cmd_args import parse_args
from src.agg_runs import agg_runs
from src.config import load_cfg, set_out_dir, set_run_dir, dump_cfg
from src.logger import setup_printing
from src.dataset.loader import load_data_and_create_dataloader
from src.model import GRAPH_MODELS
from src.model.factory import create_model
from src.train.create_trainer import create_trainer
from src.train.optim import create_criterion, create_optimizer, create_scheduler
from src.train.task import create_task
from src.utils.seed import seed_everything
from src.utils.device import auto_select_device
from src.utils.neptune import create_neptune_run_writer, write_cfg_to_neptune



def run_seed(args):
    cfg_file, out_dir, seed, is_train = args
    
    cfg = load_cfg(cfg_file)
    cfg.seed = seed
    cfg.out_dir = out_dir
    set_run_dir(cfg.out_dir, seed)
    setup_printing(logging.INFO)
    seed_everything(seed)
    auto_select_device()
    logging.info(cfg)
    
    loaders, scaler = load_data_and_create_dataloader()
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
    
    if cfg.neptune_each_run:
        neptune_writer = create_neptune_run_writer(seed)
        write_cfg_to_neptune(cfg, neptune_writer)
    else:
        neptune_writer = None
    
    trainer = create_trainer(
        model, 
        optimizer=optimizer, 
        neptune_writer=neptune_writer,
        scheduler=scheduler, 
        criterion=criterion, 
        scaler=scaler,
        task=task,
    )
    
    if is_train:
        trainer.fit(loaders)
    else:
        trainer.evaluate(loaders)


def run_single(args):    
    print(f"Running {args.cfg_file} ...")
    
    cfg = load_cfg(args.cfg_file)
    set_out_dir(cfg.out_dir, args.cfg_file)
    dump_cfg(cfg)

    if args.repeat > 1 and args.parallelseed and cfg.device != 'cpu':
        import multiprocessing
        # Repeat for different random seeds in parallel
        pool = multiprocessing.Pool()
        pool.map(run_seed, [(args.cfg_file, cfg.out_dir, i) for i in range(cfg.seed, cfg.seed + args.repeat)])
        pool.close()
        pool.join()
    else:
        num_runs = 0
        seed = cfg.seed
        while num_runs < args.repeat:
            try:
                run_seed((args.cfg_file, cfg.out_dir, seed, args.train))
                num_runs += 1
            except ValueError as e:
                logging.error(f"Random seed {seed} discarded. {e}")
                exit(1)
            seed = seed + 1
    
    # Aggregate results from different seeds
    best_metric = agg_runs(cfg.out_dir, cfg.metric_best)     
    
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    run_single(args)
