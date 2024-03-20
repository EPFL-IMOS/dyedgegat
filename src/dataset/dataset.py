import logging
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import dense_to_sparse

from .helper import get_discountinous_sequences, split_indices, get_sequences
from ..config import cfg


SCALER_DICT = {
    'minmax': MinMaxScaler,
    'standard': StandardScaler
}


def get_graph_sequences(
    X, Y, 
    edge_index, labels, 
    var_start_idx,
    aug_ocvar_on_node=False, 
    map_ocvar=False, 
    faults=None
):
    """Prepare list of sliding window data in graph format 
    Args:
    """
    X = X.astype(np.float32) # (n_samples, seq_len, n_nodes)
    Y = Y.astype(np.float32) # (n_samples, 1, n_nodes)
    labels = labels.astype(np.int32).reshape(-1, 1) # (n_samples, 1)
    if faults is not None:
        faults = faults.astype(np.int32).reshape(-1, 1) # (n_samples, 1)
    
    # n_nodes = X.shape[2]
    
    graph_list = []
    for i in range(len(X)):
        xi = torch.from_numpy(X[i].T) # [n_nodes, seq_len]
        yi = torch.from_numpy(Y[i].T) # [n_nodes, seq_len]
        label = torch.from_numpy(labels[i])
        args = {
            'x': xi,
            'y': yi,
            'edge_index': edge_index,
            'label': label
        }
        if aug_ocvar_on_node:
            ci = xi[:var_start_idx, :]
            xi = xi[var_start_idx:, :]
            yi = yi[var_start_idx:, :]
            args['x'] = xi
            args['c'] = ci
            args['y'] = yi
        elif map_ocvar:
            ci = xi[:var_start_idx, :]
            yi = xi[var_start_idx:, :]
            args['x'] = ci
            args['y'] = yi
        if faults is not None:
            args['fault'] = torch.from_numpy(faults[i])
        g = GraphData(**args)
        graph_list.append(g)
    return graph_list


def load_dataset(dataset_dir, aug_ocvar_on_node, map_ocvar):    
    test_fault_labels = None
    
    # read convars from txt file
    convar_file_path = f'{dataset_dir}/convars.txt'
    with open(convar_file_path, 'r') as file:
        conv_vars = file.read().splitlines()
    
    if 'toy' in dataset_dir:
        train_file_path = f'{dataset_dir}/{cfg.dataset.train_file}'
        test_file_path = f'{dataset_dir}/{cfg.dataset.test_file}'
        df_train_normal =  pd.read_csv(train_file_path, index_col=0)
        df_test =  pd.read_csv(test_file_path, index_col=0)
        
        columns = df_train_normal.columns
        mv_vars = [c for c in columns if c not in conv_vars+['labels']]
        oc_vars = conv_vars
        
        train_labels = np.array(df_train_normal['labels']).astype(np.bool8)
        test_labels = np.array(df_test['labels']).astype(np.bool8)
        X_train = np.array(df_train_normal[oc_vars+mv_vars].values).astype(np.float32)
        X_test =  np.array(df_test[oc_vars+mv_vars].values).astype(np.float32)

    elif 'pronto' in dataset_dir:
        train_file_path = f'{dataset_dir}/{cfg.dataset.train_file}'
        df_train_normal =  pd.read_csv(train_file_path)

        columns = df_train_normal.columns
        if cfg.dataset.aug_ocvar:
            if cfg.dataset.use_indep_vars:
                convar_file_path = f'{dataset_dir}/indep_vars.txt'
                with open(convar_file_path, 'r') as file:
                    oc_vars = file.read().splitlines()
            else:
                oc_vars = conv_vars
        else:
            oc_vars = []
        # VARS is the rest of the columns
        mv_vars = [c for c in columns if c not in oc_vars+['day', 'OC', 'Fault']]
        if cfg.dataset.no_ocvar:
            oc_vars = []

        test_file_path = f'{dataset_dir}/{cfg.dataset.test_file}'
        df_test =  pd.read_csv(test_file_path)
        
        X_train = np.array(df_train_normal[oc_vars+mv_vars].values).astype(np.float32)
        logging.info("Feature order %s", oc_vars + mv_vars)
        X_test = np.array(df_test[oc_vars+mv_vars].values).astype(np.float32)
        train_labels =  np.array(df_train_normal['Fault']!='Normal').astype(np.int32)
        test_labels = np.array(df_test['Fault']!='Normal').astype(np.int32)
        # Define a dictionary to map fault names to integers
        fault_map = {'Normal': 0, 'Air blockage': 1, 'Air leakage': 2, 'Diverted flow': 3, 'Slugging': 4}
        cfg.dataset.num_classes = df_test['Fault'].nunique()
        df_test['Fault'] = df_test['Fault'].map(fault_map)
        test_fault_labels = np.array(df_test['Fault']).astype(np.int32)

    else:
        raise ValueError(f"dataset_dir {dataset_dir} not surported, should be 'run/datasets/toy' or 'run/datasets/pronto'.")

    cfg.dataset.ocvar_dim = len(oc_vars)
    
    if aug_ocvar_on_node:
        num_nodes = len(mv_vars)   
    elif map_ocvar:
        num_nodes = len(oc_vars)
    else:
        num_nodes = len(mv_vars) + len(oc_vars)

    cfg.dataset.n_nodes = num_nodes
    
    # graph structure
    init_adj = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    init_adj = torch.from_numpy(init_adj.astype(np.float32))
    edge_index, edge_feat = dense_to_sparse(init_adj)
    num_edges = len(edge_index[0])
    edge_feat = edge_feat.reshape(-1, 1)
    
    cfg.dataset.n_edges = num_edges
    
    return X_train, train_labels, X_test, test_labels, test_fault_labels, edge_index


def prepare_countinous_graph_list(
    X_train, train_labels,
    X_test, test_labels, test_fault_labels,
    edge_index, train_test_split,
    var_start_idx, aug_ocvar_on_node, map_ocvar,
    normalize, scaler,
    win_size, horizon
):
    """Return a list of graph data for training, validation and testing from continous sequences of data 
    Test data split by train_test_split chronologically.
    """
    graph_datasets = []
    
    # train and eval
    num_timestamps_train = X_train.shape[0]
    idx_start = 0
    indices = np.arange(num_timestamps_train)
    
    # train val split
    for split_ratio in train_test_split:
        idx_split = round(num_timestamps_train * split_ratio)
        idx_end = min(idx_start+idx_split, num_timestamps_train)
        
        # raw data split
        data = X_train[indices[idx_start:idx_end]]
        label = train_labels[indices[idx_start:idx_end]]
        if idx_start == 0:
            scaler.fit(data)

        # normalize data split
        if normalize:
            data = scaler.transform(data)
        
        # get sequence data
        X_seq, Y_seq, Label_seq, _ = get_sequences(
            data, label, 
            win_size=win_size,
            horizon=horizon
        )
        
        # get graph sequence data
        graph_list = get_graph_sequences(
            X=X_seq, 
            Y=Y_seq,
            edge_index=edge_index,
            labels=Label_seq,
            var_start_idx=var_start_idx,
            map_ocvar=map_ocvar,
            aug_ocvar_on_node=aug_ocvar_on_node,
        )
        graph_datasets.append(graph_list)
        idx_start = idx_end

    # test
    # normalize test
    if normalize:
        X_test = scaler.transform(X_test)
        
    # get sequence data
    X_seq_test, Y_seq_test, Label_seq_test, Fault_seq_test = get_sequences(
        X_test, test_labels, 
        fault_labels=test_fault_labels, 
        win_size=win_size, horizon=horizon                                             
    )

    # get graph sequence data
    graph_list_test = get_graph_sequences(
        X=X_seq_test, 
        Y=Y_seq_test,
        edge_index=edge_index,
        labels=Label_seq_test,
        faults=Fault_seq_test,
        var_start_idx=var_start_idx,
        aug_ocvar_on_node=aug_ocvar_on_node,
        map_ocvar=map_ocvar,
    )
    graph_datasets.append(graph_list_test)
    return graph_datasets


def prepare_discountinous_graph_list(
    X_train, train_labels,
    X_test, test_labels, test_fault_labels,
    edge_index, 
    var_start_idx, aug_ocvar_on_node, map_ocvar,
    normalize, scaler,
    win_size, horizon
):
    """Return a list of graph data for training, validation and testing from discontinous sequences of data 
    (e.g. for the Pronto dataset the test data contains multiple faults). Test data contains part of healthy samples 
    concatenated with fault samples.
    """
    graph_datasets = []
    # get sequence data
    X_seq, Y_seq, Label_seq, _ = get_discountinous_sequences(
        X_train, 
        train_labels, 
        win_size=win_size,
        horizon=horizon,
        stride=cfg.dataset.stride,
    )
    train_indices, test_indices = split_indices(X_seq.shape[0], 
                                                test_size=cfg.dataset.test_split, 
                                                ws=win_size)
    
    X_train_seq = X_seq[train_indices]
    Y_train_seq = Y_seq[train_indices]
    X_test_seq = X_seq[test_indices]
    Y_test_seq = Y_seq[test_indices]
    Label_train_seq = Label_seq[train_indices]
    Label_test_seq = Label_seq[test_indices]
    
    # normalize data split
    if normalize:
        scaler.fit(X_train_seq[:, 0, :])

        # loop over time steps
        for i in range(X_train_seq.shape[1]):
            X_train_seq[:, i, :] = scaler.transform(X_train_seq[:, i, :])
        for i in range(Y_train_seq.shape[1]):
            Y_train_seq[:, i, :] = scaler.transform(Y_train_seq[:, i, :])
            
        for i in range(X_test_seq.shape[1]):
            X_test_seq[:, i, :] = scaler.transform(X_test_seq[:, i, :])
        for i in range(Y_test_seq.shape[1]):
            Y_test_seq[:, i, :] = scaler.transform(Y_test_seq[:, i, :])
    
    # get graph sequence data
    graph_list_train = get_graph_sequences(
        X=X_train_seq, 
        Y=Y_train_seq,
        edge_index=edge_index,
        labels=Label_train_seq,
        var_start_idx=var_start_idx,
        map_ocvar=map_ocvar,
        aug_ocvar_on_node=aug_ocvar_on_node,
    )
    
    graph_list_train, graph_list_eval = train_test_split(graph_list_train, shuffle=True, random_state=cfg.seed, test_size=cfg.dataset.val_split)
    
    graph_datasets.append(graph_list_train)
    graph_datasets.append(graph_list_eval)
    
    X_seq_f, Y_seq_f, Label_seq_f, Fault_seq_f = get_discountinous_sequences(
        X_test, 
        test_labels, 
        win_size=win_size,
        horizon=horizon,
        fault_labels=test_fault_labels,
        stride=cfg.dataset.stride,
    )
    
    if normalize:
        # loop over time steps
        for i in range(X_seq_f.shape[1]):
            X_seq_f[:, i, :] = scaler.transform(X_seq_f[:, i, :])
        for i in range(Y_seq_f.shape[1]):
            Y_seq_f[:, i, :] = scaler.transform(Y_seq_f[:, i, :])

    graph_list_test = get_graph_sequences(
        X=np.concatenate((X_test_seq, X_seq_f), axis=0), 
        Y=np.concatenate((Y_test_seq, Y_seq_f), axis=0),
        edge_index=edge_index,
        labels=np.concatenate((Label_test_seq, Label_seq_f), axis=0),
        faults=np.concatenate((Label_test_seq, Fault_seq_f), axis=0),
        var_start_idx=var_start_idx,
        map_ocvar=map_ocvar,
        aug_ocvar_on_node=aug_ocvar_on_node,
    )

    graph_datasets.append(graph_list_test)
    test_labels = np.concatenate([Label_seq_f, Label_seq[test_indices]])
    return graph_datasets


def load_data(
    dataset_dir, 
    win_size=32,
    horizon=1,
    aug_ocvar_on_node=False,
    map_ocvar=False,
    return_scaler=False,
):
    
    normalize = cfg.dataset.normalize
    dataset_dir = f'run/{dataset_dir}'
    
    X_train, train_labels, X_test, test_labels, test_fault_labels, edge_index = load_dataset(dataset_dir, aug_ocvar_on_node, map_ocvar)

    scaler = SCALER_DICT[cfg.dataset.scaler_type]()
    
    graph_datasets = []

    var_start_idx = cfg.dataset.ocvar_dim
    if 'pronto' in dataset_dir:
        graph_datasets = prepare_discountinous_graph_list(
            X_train, train_labels,
            X_test, test_labels, test_fault_labels,
            edge_index, 
            var_start_idx, aug_ocvar_on_node, map_ocvar,
            normalize, scaler,
            win_size, horizon
        )
        
    else:
        train_test_split = [1-cfg.dataset.test_split, cfg.dataset.test_split]
        graph_datasets = prepare_countinous_graph_list(
            X_train, train_labels,
            X_test, test_labels, test_fault_labels,
            edge_index, train_test_split,
            var_start_idx, aug_ocvar_on_node, map_ocvar,
            normalize, scaler,
            win_size, horizon
        )

    n_train_graphs = len(graph_datasets[0])
    n_val_graphs = len(graph_datasets[1])
    n_test_graphs = len(graph_datasets[2])
    num_graphs = n_train_graphs + n_val_graphs + n_test_graphs
    
    cfg.dataset.n_train = n_train_graphs
    cfg.dataset.n_val = n_val_graphs
    cfg.dataset.n_test = n_test_graphs
    
    d_info = '\n'
    d_info += f'Loaded data from {dataset_dir}\n'
    d_info += f'------------ Basic -----------\n'
    d_info += f'# Samples: {num_graphs}\n'
    d_info += f'# Node: {cfg.dataset.n_nodes}\n'
    d_info += f'# Edges: {cfg.dataset.n_edges}\n'
    d_info += f'# Sequence Length: {win_size}\n'
    
    d_info += f'------------ Fault ratio -----------\n'
    d_info += f'# Ratio of fault samples train: {sum(train_labels)/len(train_labels):.3f}\n'
    d_info += f'# Ratio of fault samples test: {sum(test_labels)/len(test_labels):.3f}\n'

    if aug_ocvar_on_node:
        d_info += f'------------ Control -----------\n'
        d_info += f'Control feature size: {cfg.dataset.ocvar_dim}\n'

    if normalize:
        d_info += f'------------ Scaling -----------\n'
        if cfg.dataset.scaler_type == 'minmax':
            d_info += f'{np.array2string(scaler.data_range_, precision=2,)}\n'
        elif cfg.dataset.scaler_type == 'standard':
            d_info += f'{np.array2string(scaler.mean_, precision=2,)}\n'
    
    d_info += f'------------ Split -----------\n'
    d_info += f'   Train: {n_train_graphs}/{num_graphs}\n'
    d_info += f'Validate: {n_val_graphs}/{num_graphs}\n'
    d_info += f'    Test: {n_test_graphs}/{num_graphs}\n'
    logging.info(d_info)
    
    if return_scaler:
        return graph_datasets, scaler
    else:
        return graph_datasets


def create_data_loaders(datasets, batch_size=16):
    d_train, d_val, d_test = datasets

    train_loader = DataLoader(d_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(d_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(d_test, batch_size=batch_size, shuffle=False)      
    return train_loader, val_loader, test_loader
