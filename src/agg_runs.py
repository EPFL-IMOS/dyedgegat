import json
import logging
import os
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from .config import cfg
from .utils.io import (
    dict_list_to_json,
    json_to_dict_list, 
    dict_list_to_tb,
    makedirs, 
    makedirs_rm_exist
)
from .utils.neptune import create_neptune_run_writer, get_run_id, write_agg_results_to_neptune, write_cfg_to_neptune
from .utils.seed import is_seed


def is_split(s):
    if s in ['train', 'eval', 'test']:
        return True
    else:
        return False


def join_list(l1, l2):
    assert len(l1) == len(l2), \
        'Results with different seeds must have the save format'
    for i in range(len(l1)):
        l1[i] += l2[i]
    return l1


def dict_to_json(dict, fname):
    """
    Dump a Python dictionary to JSON file

    Args:
        dict (dict): Python dictionary
        fname (str): Output file name

    """
    with open(fname, 'a') as f:
        json.dump(dict, f, indent=4, sort_keys=True)
        f.write('\n')


def agg_dict_list(dict_list):
    """
    Aggregate a list of dictionaries: mean + std
    Args:
        dict_list: list of dictionaries

    """
    dict_agg = {'epoch': dict_list[0].get('epoch')}
    for key in dict_list[0]:
        if key != 'epoch':
            try:
                value = np.array([dict[key] for dict in dict_list])
                dict_agg[key] = np.mean(value).round(cfg.get('round', 7))
                dict_agg[f'{key}_std'] = np.std(value).round(cfg.get('round', 7))
            except ValueError as e:
                print(e)
                
    return dict_agg


def write_result_to_xlsx(results_list, results_agg, file_path):
    '''
    Summarize a dictionary of statistics to a xlsx file
    '''
    df_all = pd.DataFrame(results_list['test'], index=range(len(results_list['test'])))    
    current_case = cfg.out_dir.split('/')[-1]
    df_all['case'] = current_case
    df_all.sort_index(axis=1, inplace=True)
    df_all['time_stamp'] = pd.Timestamp.now()
    df_best = pd.DataFrame(results_agg['test'], index=[current_case])
    df_best['time_stamp'] = pd.Timestamp.now()  
    # df = df_all.merge(df_best, how='outer')
    
    is_exist = os.path.exists(file_path)
    mode = 'a' if is_exist else 'w'
    if_sheet_exists = 'overlay' if is_exist else None
    with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:  
        header = not is_exist
        # df_all_old = pd.read_excel(writer, sheet_name=cfg.case) if is_exist else None
        start_row = 0 if not is_exist else writer.sheets[cfg.case].max_row
        df_all.to_excel(writer, sheet_name=cfg.case, startrow=start_row, header=header)        
        start_row = 0 if not is_exist else writer.sheets['all'].max_row
        df_best.to_excel(writer, sheet_name='all', startrow=start_row, header=header)


def agg_runs(out_dir, metric_best='auto'):
    r"""
    Aggregate over different random seeds of a single experiment
    Args:
        out_dir (str): Directory of the results, containing 1 experiment
        metric_best (str, optional): The metric for selecting the best
        validation performance.
        Options:
            auto,
            mae,
            mse,
            rmse,
    """
    results = {'train': None, 'eval': None, 'test': None}
    results_best = {'train': None, 'eval': None, 'test': None}

    metric = 'mae' if metric_best == 'auto' else metric_best

    for seed in os.listdir(out_dir):
        if is_seed(seed):
            dir_seed = os.path.join(out_dir, seed)

            split = 'eval'
            if split in os.listdir(dir_seed):
                dir_split = os.path.join(dir_seed, split)
                makedirs(dir_split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                stats_list = json_to_dict_list(fname_stats)
                performance_np = np.array([stats[metric] for stats in stats_list])
                index = eval("performance_np.{}()".format(cfg.metric_agg))
                best_epoch = stats_list[index]['epoch']

            for split in os.listdir(dir_seed):
                if is_split(split):
                    dir_split = os.path.join(dir_seed, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    stats_list = json_to_dict_list(fname_stats)
                    stats_best = [stats for stats in stats_list if stats['epoch'] == best_epoch][0]
                    print(split, stats_best)
                    stats_list = [[stats] for stats in stats_list]
                    if results[split] is None:
                        results[split] = stats_list
                    else:
                        last_idx = min(len(results[split]), len(stats_list))
                        results[split] = join_list(results[split][:last_idx], stats_list[:last_idx])
                    if results_best[split] is None:
                        results_best[split] = [stats_best]
                    else:
                        results_best[split] += [stats_best]

    results = {k: v for k, v in results.items() if v is not None}  # rm None
    results_best = {k: v for k, v in results_best.items() if v is not None}  # rm None
    results_best_list = results_best.copy()

    for key in results:
        for i in range(len(results[key])):
            results[key][i] = agg_dict_list(results[key][i])
    for key in results_best:
        results_best[key] = agg_dict_list(results_best[key])
        
    # save aggregated results
    for key, value in results.items():
        dir_out = os.path.join(out_dir, 'agg', key)
        makedirs_rm_exist(dir_out)
        fname = os.path.join(dir_out, 'stats.json')
        dict_list_to_json(value, fname)

        if cfg.tensorboard_agg:
            writer = SummaryWriter(dir_out)
            dict_list_to_tb(value, writer)
            writer.close()

    for key, value in results_best.items():
        dir_out = os.path.join(out_dir, 'agg', key)
        fname = os.path.join(dir_out, 'best.json')
        dict_to_json(value, fname)

    if cfg.neptune_agg:
        run_id = get_run_id(seed)
        neptune_writer = create_neptune_run_writer(run_id)
        write_cfg_to_neptune(cfg, neptune_writer)
        write_agg_results_to_neptune(results_best, neptune_writer)
        neptune_writer.stop()
    
    logging.info('Results aggregated across runs saved in {}'.format(
        os.path.join(out_dir, 'agg')))

    # best_metric = results_best['eval'].get(metric, 1e6)
    return results_best