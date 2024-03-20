from typing import Union, List, Optional
import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, 
    precision_score, recall_score, 
    roc_auc_score, accuracy_score, 
    roc_curve, precision_recall_curve,
    average_precision_score,
    roc_curve, auc
)
import numpy.ma as ma
from numpy.lib.stride_tricks import sliding_window_view
from tadpak.evaluate import evaluate
from tadpak.pak import pak

from ..config import cfg
from ..model.baseline.grelen import get_grelen_ad_score
from ..model.baseline.usad import get_usad_ad_score
from ..model.baseline.gdn import get_gdn_ad_score


def calculate_basic_score(trues, preds, metric='mae'):
    # calculate score for each step (step is in the last column)
    if metric == 'mae':
        score = abs(trues - preds)
    elif metric == 'mse':
        score = (trues - preds) ** 2
    else:
        raise NotImplementedError
    return score.mean(axis=2)


def calculate_iqr_median(score):    
    err_iqr = np.quantile(score, 0.75, axis=0) - np.quantile(score, 0.25, axis=0)
    err_median = np.median(score, axis=0)
    return err_iqr, err_median


def scale_by_median_iqr(score: np.ndarray, err_iqr: np.ndarray, err_median: np.ndarray) -> np.ndarray:
    return abs(score - err_median) / (err_iqr + 1e-5)


def get_anomaly_score(
    trues: Union[np.ndarray, List[np.ndarray]], 
    preds: Union[np.ndarray, List[np.ndarray]],
    scale_by_err: bool=False,
    scale_by_att: bool=False,
    err_iqr=None,
    err_median=None,
    metric: str='mae',
    topk: Optional[int]=None,
    gamma: float=1.,
    atts: Optional[np.ndarray]=None,
):
    """
    Calculate an anomaly score based on either a single modality (e.g., reconstruction or forecasting)
    or a combination of multiple modalities, and with optional scaling by median and interquartile range (IQR).
    
    The function is designed to be flexible, allowing for different types of scaling and optional use of top-k nodes.
    
    The use of median and IQR for scaling, as well as the option to use top-k nodes (specifically when topk=1 and scale_by_median_iqr=True),
    is based on concepts from the Graph Deviation Network (GDN) project. For the original GDN code, refer to: 
    https://github.com/d-ailin/GDN/blob/main/evaluate.py
    
    Parameters:
        trues (Union[np.ndarray, List[np.ndarray]]): Ground truth values.
            If using a single modality, the shape is [n_samples, n_nodes, n_steps].
            If combining modalities, pass a list of np.ndarrays with corresponding shapes.
        
        preds (Union[np.ndarray, List[np.ndarray]]): Predicted values. Same shape as `trues`.

        err_iqr (Optional[np.ndarray]): Pre-computed interquartile range for anomaly score scaling.
            If None, will be calculated internally.

        err_median (Optional[np.ndarray]): Pre-computed median for anomaly score scaling.
            If None, will be calculated internally.

        scale_by_err (bool): Whether to scale the score by the median and IQR.
            Default is False.
            
        scale_by_att (bool): Whether to scale the score by the attention adjacency matrix.
            Default is False.

        topk (Optional[int]): If set, selects the top-k nodes with the highest error to average for each sample.
            When set to None, uses all nodes.
            
        gamma (float): Weight for combining multiple modalities, e.g., reconstruction and forecasting.
             Default is 1.

        atts : Optional[np.ndarray]
            Learned attention scores between nodes, representing the weight of the edges. Shape varies:
            - If it's a 3D array, then [n_samples, n_nodes, n_nodes]
            - If it's a 4D array, then [n_samples, n_nodes, n_nodes, n_heads]
            These scores can be viewed as an adjacency matrix and are used to scale the anomaly score.
            Shape can be either [n_samples, n_nodes, n_nodes] or [n_samples, n_nodes, n_nodes, n_heads].
    
    Returns:
        np.ndarray: Anomaly score for each sample.
        np.ndarray: Interquartile range used for scaling.
        np.ndarray: Median used for scaling.
    """
    
    # If we have a tuple of trues and preds, it means we are combining scores
    if isinstance(trues, tuple) and isinstance(preds, tuple):
        # decompose into reconstruction and forecast components
        recon_true, fc_true = trues # [n_samples, n_nodes, n_steps],  [n_samples, n_nodes, 1]
        recon_pred, fc_pred = preds

        # Compute the individual scores
        score_recon = calculate_basic_score(recon_true, recon_pred, metric=metric)
        score_fc = calculate_basic_score(fc_true, fc_pred, metric=metric)
        
        # Combine them
        score = score_recon + gamma * score_fc
    else:
        score = calculate_basic_score(trues, preds, metric=metric)

    if scale_by_att and atts is not None:
        score = scale_by_attention(score, atts)

    # scale by single node's median and IQR
    if scale_by_err:
        if err_iqr is None or err_median is None:
            err_iqr, err_median = calculate_iqr_median(score)
        # scaling both val and test scores by the same median and IQR from val set
        score = scale_by_median_iqr(score, err_iqr, err_median)
    
    # smoothing scores with sliding window similar as in GDN
    if cfg.task.anomaly_score_sw > 1:
        score = smooth_score(score, cfg.task.anomaly_score_sw)
    
    # top-k nodes if specified
    if topk is not None: 
        n_nodes = trues.shape[1]
        # select top k errors for each graph (GDN)
        topk_indices = np.argpartition(score, range(n_nodes-topk-1, n_nodes), axis=1)[:, -topk:]
        score = np.sum(np.take_along_axis(score, topk_indices, axis=1), axis=1)
    else:
        # mean over all nodes
        score = score.mean(axis=1)
    
    return score, err_iqr, err_median


def get_degree(adj):
    # adj [n_nodes, n_nodes]
    return np.sum(adj, axis=0) + np.sum(adj, axis=1)


def scale_by_attention(score: np.ndarray, atts: np.ndarray) -> np.ndarray:
    if atts.ndim == 4:
        degree = np.asarray([[get_degree(atts[i, :, :, j]) for j in range(atts.shape[3])] for i in range(atts.shape[0])]).mean(axis=1)
    else:
        degree = np.array([get_degree(atts[i]) for i in range(atts.shape[0])])
    return score * degree


def smooth_score(score: np.ndarray, sliding_window: int):
    # mask to select smoothed scores
    if sliding_window > 1:
        if score.ndim == 1:
            score = np.average(sliding_window_view(score, window_shape=sliding_window), axis=1)
        else:
            score = np.average(sliding_window_view(score, window_shape=(sliding_window, 1)), axis=2)
    return score


def find_nearst_larger_idx(arr, target):
    idx = np.where(arr >= target)[0]
    if len(idx) == 0:
        return np.inf
    else:
        return arr[idx[0]]


def get_detection_delay(pred_labels: np.ndarray, true_labels: np.ndarray, timesteps: int):
    # Initialize a list to store the delay for each fault
    delays = []

    # Initialize a variable to keep track of the start of the fault
    fault_start = 0
    fault_end = 0

    is_previous_fault = False

    n_faults = 0

    # Iterate over the labels
    for i in range(timesteps, len(true_labels)-1):
        # if currently is a fault and previous was not
        if true_labels[i] == 1 and not is_previous_fault:
            fault_start = i
            n_faults += 1

        # If a fault is predicted and a fault has started
        is_pred_fault = pred_labels[i-timesteps:i].sum() == timesteps
        if is_pred_fault and (i > fault_start) and (len(delays) < n_faults):
            # Calculate the delay and add it to the list
            delay = i - fault_start
            delays.append(delay)
            # print("DETECTED: ", delay)
        
        # if currently is not a fault and previous was
        if true_labels[i] == 0 and is_previous_fault:
            fault_end = i

            # If a fault is not predicted
            if not is_pred_fault and (len(delays) < n_faults):
                # Calculate the delay and add it to the list
                delay = fault_end - fault_start
                # print("MAX DELAY: ", delay)
                delays.append(delay)
        
        is_previous_fault = true_labels[i] == 1
    
    # If the last fault is not detected
    if len(delays) < n_faults:
        # last timestamp is a fault
        fault_end = len(true_labels)
        delay = fault_end - fault_start
        delays.append(delay)
        # print("MAX DELAY: ", delay)
            
    return delays


def get_auc_per_class(
    true_labels: np.ndarray, 
    pred_labels: np.ndarray,
    fault_classes: np.ndarray,
):
    result = {}
    fault_class_labels = [i for i in np.unique(fault_classes) if i != 0]
    # healthy class
    h_idx = np.where(fault_classes == 0)[0]
    true_labels_h = true_labels[h_idx]
    pred_labels_h = pred_labels[h_idx]

    if 'pronto' in cfg.dataset.dir:
        fault_map = {1: 'Air blockage', 2: 'Air leakage', 3: 'Diverted flow', 4: 'Slugging'}
    elif 'metallicadour' in cfg.dataset.dir:
        fault_map = {0: 'healthy', 1: 'broken_tooth', 2: 'surface_damage', 3: 'flack_damage'}
    elif 'toy' in cfg.dataset.dir:
        fault_map = {i: f'Node {i}' for i in range(1, 6)}

    for c in fault_class_labels:
        # find all instances of class c
        idx = np.where(fault_classes == c)[0]
        true_labels_c = true_labels[idx]
        pred_labels_c = pred_labels[idx]
        t_label = np.concatenate([true_labels_h, true_labels_c])
        p_label = np.concatenate([pred_labels_h, pred_labels_c])
        # weights = compute_sample_weight('balanced', t_label)
        # calculate auc for class c
        auc_lab = roc_auc_score(t_label, p_label)
        result[f'{fault_map[c]}'] = auc_lab
    result['auc_class_avg'] = np.mean(list(result.values()))
    # get result without the slugging class
    if 'pronto' in cfg.dataset.dir:
        result['auc_class_avg_no_slugging'] = np.mean([v for k, v in result.items() if k != 'Slugging' and k != 'auc_class_avg'])
    return result


def get_detection_metrics(true_labels, pred_labels, score, fault_labels, threshold):
    try:
        fpr, tpr, _ = roc_curve(true_labels, score)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        roc_auc = 0.
    
    pa = average_precision_score(true_labels, score)
    
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, normalize='all').ravel()
    
    ambiguity = 1 - 2 * abs(roc_auc - 0.5)
    
    out = {
        'ambiguity': round(ambiguity, cfg.round),
        'accuracy': round(accuracy_score(true_labels, pred_labels), cfg.round),
        'precision': round(precision_score(true_labels, pred_labels, zero_division=1), cfg.round),
        'recall': round(recall_score(true_labels, pred_labels), cfg.round),
        'f1': round(f1_score(true_labels, pred_labels), cfg.round),
        'auc': round(roc_auc, cfg.round),
        'pa': round(pa, cfg.round),
        'tn': round(tn, cfg.round),
        'fp': round(fp, cfg.round),
        'fn': round(fn, cfg.round),
        'tp': round(tp, cfg.round),
        'tp_rate': round(tp / (tp + fn + 1e-8), cfg.round),
        'fp_rate': round(fp /( fp + tn + 1e-8), cfg.round),
        'fpr': fpr, 
        'tpr': tpr,
    }

    # k: PA%K threshold. If K = 0, it is equal to the F1PA and if K = 100, it is equal to the F1.
    try:
        pak_results = evaluate(score, true_labels, k=cfg.task.anomaly_score_pak)
        out.update(pak_results)
        score_pa = pak(score, true_labels, threshold, k=0)
        auc_pa = roc_auc_score(true_labels, score_pa)
        out['auc_pa'] = round(auc_pa, cfg.round)
        score_pak = pak(score, true_labels, threshold, k=cfg.task.anomaly_score_pak)
        auc_pak = roc_auc_score(true_labels, score_pak)
        out['auc_pak'] = round(auc_pak, cfg.round)
    except ValueError:
        out['auc_pa'] = 0.
        out['auc_pak'] = 0.
    
    if fault_labels is not None:
        try:
            acc_per_class = get_auc_per_class(true_labels, score, fault_classes=fault_labels)
            # Merge the two dictionaries
            out.update(acc_per_class)
        except ValueError:
            pass

    return out
    

def get_detection_threshold(mode, val_score, test_score, true_labels):
    # compute and return threshold
    if mode == 'val':
        if cfg.task.anomaly_score_thresh < 1.0:
            threshold = np.quantile(val_score, cfg.task.anomaly_score_thresh)
        else:
            threshold = cfg.task.anomaly_score_thresh * np.max(val_score)
    
    elif mode == 'best_auc':
        # use roc to find best threshold
        fpr, tpr, thresholds = roc_curve(true_labels, test_score)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
    
    elif mode == 'best_prauc':
        # use roc to find best threshold
        fpr, tpr, thresholds = precision_recall_curve(true_labels, test_score)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
    
    else:
        raise ValueError(f'Unknown anomaly score threshold mode: {cfg.task.anomaly_score_thresh_mode}, should be one of: val, best.')
    return threshold


def get_prediction_scores(
    anomaly_score_func, 
    val_preds, val_trues, 
    test_preds, test_trues, 
    true_labels,
    metric: str='mae',
    adj_val: Optional[np.ndarray]=None, 
    adj_test: Optional[np.ndarray]=None,
    gamma: float=1.0,
    fault_labels: Optional[np.ndarray]=None,
    return_scores=False,
):
    
    ad_arg_mapping = {
        'base': {},
        'node_scaled': {'scale_by_err': True},
        'node_scaled_topk': {'scale_by_err': True, 'topk': cfg.task.anomaly_score_topk},
        'node_scaled_combined': {'scale': True},
        'att_scaled': {'scale_by_att': True},
        'att_scaled_topk': {'scale_by_att': True, 'topk': cfg.task.anomaly_score_topk},
        'att_node_scaled': {'scale_by_err': True, 'scale_by_att': True},
        'gdn': {},
        'grelen_ad': {},
        'usad': {},
    }
    
    # Check if the given anomaly_score_func is supported
    if anomaly_score_func in ad_arg_mapping:
        
        args = ad_arg_mapping[anomaly_score_func]
        if anomaly_score_func == 'grelen_ad':
            val_score = get_grelen_ad_score(adj_val)
            test_score = get_grelen_ad_score(adj_test)
            # because grelen has a smoothing of 2, we need to remove the first score
            true_labels = true_labels[1:]
            fault_labels = fault_labels[1:] if fault_labels is not None else None
        elif anomaly_score_func == 'usad':
            val_score = get_usad_ad_score(trues=val_trues, preds=val_preds)
            test_score = get_usad_ad_score(trues=test_trues, preds=test_preds)
        elif anomaly_score_func == 'gdn_ad':
            cfg.task.anomaly_score_sw = 1 # smoothed inside the get_gdn_ad_score function
            cfg.task.anomaly_score_thresh_mode = 'val'
            cfg.task.anomaly_score_thresh = 1.0
            val_score = get_gdn_ad_score(trues=val_trues, preds=val_preds)
            test_score = get_gdn_ad_score(trues=test_trues, preds=test_preds)
        else:
            # Validation score
            val_score, err_iqr, err_median = get_anomaly_score(
                val_trues, val_preds, metric=metric, 
                scale_by_err=args.get('scale_by_err', False),
                scale_by_att=args.get('scale_by_att', False),
                err_iqr=None, err_median=None,
                topk=args.get('topk', None), gamma=gamma, atts=adj_val,
            )
            # Test score
            test_score, _, _ = get_anomaly_score(
                test_trues, test_preds, metric=metric,
                scale_by_err=args.get('scale_by_err', False),
                scale_by_att=args.get('scale_by_att', False),
                err_iqr=err_iqr, err_median=err_median,
                topk=args.get('topk', None), gamma=gamma, atts=adj_test,
            )

    else:
        raise ValueError(f'Unknown anomaly score function: {anomaly_score_func}. '+
                         'Should be one of: base, node_scaled, node_scaled_combined, node_scaled_topk, att_scaled, gdn, grelen, usad.')
    
    threshold = get_detection_threshold(cfg.task.anomaly_score_thresh_mode, val_score, test_score, true_labels)
    pred_labels = (test_score > threshold)
    
    # mask the first and last n steps due to node score smoothing
    if cfg.task.anomaly_score_sw > 1:
        score_mask = np.ones_like(true_labels)
        score_mask[:cfg.task.anomaly_score_sw//2] = 0
        score_mask[-(cfg.task.anomaly_score_sw//2):] = 0
        score_mask = ma.make_mask(score_mask)
        
        true_labels = true_labels[score_mask]
        fault_labels = fault_labels[score_mask] if fault_labels is not None else None
    
    # Assert true_labels and pred_labels have the same length
    assert len(true_labels) == len(pred_labels), f'Length of true_labels ({len(true_labels)}) and pred_labels ({len(pred_labels)}) do not match.'
    
    out = get_detection_metrics(true_labels, pred_labels, test_score, fault_labels, threshold)
    
    # Delay related evaluation metrics
    delays = get_detection_delay(pred_labels, true_labels, timesteps=cfg.task.detection_delay_ts)
    
    threshold_best_auc = get_detection_threshold('best_auc', val_score, test_score, true_labels)
    pred_labels_best_auc = (test_score > threshold_best_auc)
    delays_best_auc = get_detection_delay(pred_labels_best_auc, true_labels, timesteps=cfg.task.detection_delay_ts)
    
    threshold_best_pr = get_detection_threshold('best_prauc', val_score, test_score, true_labels)
    pred_labels_best_pr = (test_score > threshold_best_pr)
    delay_best_pr = get_detection_delay(pred_labels_best_pr, true_labels, timesteps=cfg.task.detection_delay_ts)
    
    out.update({
        'threshold': round(threshold, cfg.round),
        'delay': round(np.nanmean(delays), cfg.round),
        'delays_best_auc': round(np.nanmean(delays_best_auc), cfg.round),
        'delay_best_pr': round(np.nanmean(delay_best_pr), cfg.round),
    })    
    if return_scores:
        return out, (val_score, test_score)
    return out


def select_time_steps(values, n_steps, reverse=False):
    """ Select n time steps from the a tensor or a tuple of tensors. """
    def slice_tensor(tensor):
        sliced = tensor[..., -n_steps:] if reverse else tensor[..., :n_steps]
        # Ensure 3D shape by reshaping if necessary
        while len(sliced.shape) < 3:
            sliced = np.expand_dims(sliced, -1)
        return sliced
    
    if isinstance(values, tuple):
        sliced_values = tuple(slice_tensor(slice_tensor(tensor)) for tensor in values)
    else:
        sliced_values = slice_tensor(slice_tensor(values))

    return sliced_values


def get_best_prediction_scores(total_steps, metric='auc', return_scores=False, **kwargs):
    if metric in ['all', 'last', 'first']:
        n_steps = total_steps if metric == 'all' else 1
        reverse = metric == 'last'
        kwargs['return_scores'] = return_scores
        for key in ['val_preds', 'val_trues', 'test_preds', 'test_trues']:
            kwargs[key] = select_time_steps(kwargs[key], n_steps, reverse)
        out = get_prediction_scores(**kwargs)
        out['n_steps'] = n_steps
        return out
    else:
        all_outs = []
        for i in range(total_steps):
            for key in ['val_preds', 'val_trues', 'test_preds', 'test_trues']:
                kwargs[key] = select_time_steps(kwargs[key], i+1)
            kwargs['return_scores'] = return_scores
            all_outs.append(get_prediction_scores(**kwargs))
            out['n_steps'] = i + 1
        all_metrics = np.array([out[metric] for out in all_outs])
        idx = np.argmax(all_metrics)
        return all_outs[idx]
