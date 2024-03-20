import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj

from ..config import cfg


def plot_roc_auc(fpr, tpr, auc, figsize=(5, 5), color='skyblue', alpha=0.4):
    """
    Plots ROC curve and area under the ROC curve.
    
    Parameters:
    - y_true: Ground truth binary labels.
    - y_score: Target scores, can either be probability estimates or non-thresholded measure of decisions.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    ax.fill_between(fpr, tpr, color=color, alpha=alpha, label="Area under curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Receiver Operating Characteristic', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True)
    return fig


def plot_confusion_matrix(tn, fp, fn, tp, 
                          labels=['Normal', 'Anomaly'], 
                          title='Confusion matrix', 
                          figsize=(5, 5)):
    """
    Plot confusion matrix
    """
    conf_mat = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(conf_mat, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                xticklabels=labels,
                yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return fig


@torch.no_grad()
def get_feauture_attention(batch, model):
    model.eval()
    # batch.x [b, nf, k], b: batch size, nf: number of features, k: sequence length
    f_edge_index = batch.edge_index
    
    c_temp = None
    if model.aug_control:
        c_temp = model.control_encoder(batch.c, batch=batch.batch)

    # temporal node representation
    x_temp = model.node_encoder(batch.x, batch=batch.batch, c=c_temp)
    
    f_edge_index, f_edge_attr = model.feat_edge_layer(
        x_temp, 
        edge_index=f_edge_index, 
        batch=batch.batch
    )
    A = to_dense_adj(f_edge_index.detach(), edge_attr=f_edge_attr.detach(), batch=batch.batch).cpu().numpy()
    return A


@torch.no_grad()
def get_graph_attention(batch, model):
    model.eval()
    # batch.x [b, nf, k], b: batch size, nf: number of features, k: sequence length
    
    c_temp = None
    if model.aug_control:
        c_temp = model.control_encoder(batch.c, batch=batch.batch)

    # temporal node representation
    x_temp = model.node_encoder(batch.x, batch=batch.batch, c=c_temp)
    f_edge_index, aggr_edge_attr = model.learn_graph(x_temp, batch)
    
    A = to_dense_adj(f_edge_index.detach(), edge_attr=aggr_edge_attr.detach(), batch=batch.batch)
    if A.ndim == 4:
        A = A.squeeze(3)
    A = A.cpu().numpy()
    return A


def visualize_attention(adj):
    ai = adj.reshape(cfg.dataset.n_nodes, cfg.dataset.n_nodes)
    fig, _ = plt.subplots(1, 1, figsize=(6, 5))
    mask = ai==0.
    sns.heatmap(ai, cmap="Blues", vmin=0., square=True, mask=mask)
    return fig


def plot_node_forecasting(pred, true):
    n_nodes = cfg.model.output_dim if cfg.task.train_type == 'mapping' else cfg.dataset.n_nodes
    n_cols = int(np.ceil(np.sqrt(n_nodes)))
    n_rows = int(np.ceil(n_nodes / n_cols))
    f, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    for idx in range(n_nodes):
        axes[idx//n_cols, idx%n_cols].plot(true[idx:n_nodes*500:n_nodes], label='true')
        axes[idx//n_cols, idx%n_cols].plot(pred[idx:n_nodes*500:n_nodes], label='pred')
        axes[idx//n_cols, idx%n_cols].legend()
    return f
