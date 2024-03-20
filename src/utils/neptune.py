"""
Experiment tracking with Neptune
"""
import neptune

from ..config import cfg
from ..utils.credential import (
    NEPTUNE_PROJECT_NAME, 
    NEPTUNE_PROJECT_CRED
)


CASE_MAPPING = {
    'test': 'TE',
    'pronto': 'PRO',
    'toy': 'TOY',
}


def get_run_id():    
    if '/' in cfg.case:
            case_name = cfg.case.split('/')[-1]
    else:
        case_name = cfg.case
    case_name = CASE_MAPPING[case_name]
    fname = cfg.fname.replace('=', '')
    ad_score_func = "".join([c[0] for c in cfg.task.anomaly_score_func.split('_')])
    run_id = f'{case_name}-{fname}-{case_name}-{ad_score_func}'
    if len(run_id) > 35:
        run_id = run_id.replace('-', '')
    run_id = f'{run_id}'
    return run_id


def create_neptune_run_writer(seed):
    id = 'Untitled'
    try:
        id = get_run_id(seed)
        neptune_writer = get_neptune_run(id)
    except Exception as e:
        neptune_writer = neptune.init_run(
            project=NEPTUNE_PROJECT_NAME,
            api_token=NEPTUNE_PROJECT_CRED,
            source_files=['src/'],
            custom_run_id=id
        )  # defined in the .env file

    return neptune_writer


def get_neptune_run(run_id):
    run = neptune.init_run(
        project=NEPTUNE_PROJECT_NAME,
        api_token=NEPTUNE_PROJECT_CRED,
        with_id=run_id
    )  # defined in the .env file
    return run


def write_cfg_to_neptune(cfg, neptune_writer):
    for key_dict, value in cfg.items():
        try:
            if isinstance(key_dict, dict):
                for k, v in key_dict.items():
                    if isinstance(v, list):
                        neptune_writer[k].append(v)
                    else:
                        neptune_writer[k] = v
            else:
                neptune_writer[key_dict] = value
        except ValueError as e:
            print(f'Error writing {key_dict} to Neptune: {e}')


def write_agg_results_to_neptune(results, neptune_writer):
    neptune_writer['epoch'] = results['train']['epoch']
    neptune_writer['params'] = results['train']['params']
    for split, split_result in results.items():
        split_result.pop('params', None)
        split_result.pop('params_std', None)
        split_result.pop('epoch', None)
        split_result.pop('epoch_std', None)
        split_result.pop('lr', None)
        split_result.pop('lr_std', None)
        split_result.pop('time_iter', None)
        split_result.pop('time_iter_std', None)
        for key, value in split_result.items():
            try:
                if isinstance(value, float) or isinstance(value, int):
                    neptune_writer[f'{split}/{key}'] = value
                elif isinstance(value, list):
                    neptune_writer[f'{split}/{key}'].append(value)
            except Exception as e:
                print(f'Error writing {key} to Neptune: {e}')


def write_test_results_to_neptune(results, neptune_writer):
    for key, value in results.items():
        try:
            if isinstance(value, float) or isinstance(value, int):
                neptune_writer[f'test/{key}'] = value
            elif isinstance(value, list):
                neptune_writer[f'test/{key}'].append(value)
        except Exception as e:
            print(f'Error writing {key} to Neptune: {e}')