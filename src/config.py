import os, stat
import shutil
import yaml

from yacs.config import CfgNode as CN


# Global config object
cfg = CN()


def set_cfg(cfg):
    r'''
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name
    :return: configuration use by the experiment.
    '''
    if cfg is None:
        return cfg

    # ----------------------------------------------------------------------- #
    # Basic options
    # ----------------------------------------------------------------------- #
    
    # Set print destination: stdout / file / both
    cfg.print = 'both'
    
    cfg.metric_agg = 'argmin'
    cfg.metric_best = 'mae'
    
    # Select device: 'cpu', 'cuda:0', 'auto'
    cfg.device = 'cpu'
    
    # Output directory
    cfg.out_dir = 'run/results'
    
    cfg.fname = ''
    
    cfg.case = ''
    
    cfg.cfg_dest = 'config.yaml'
    
    # Random seed
    cfg.seed = 0

    # If get GPU usage
    cfg.gpu_mem = False

    # Print rounding
    cfg.round = 5

    # Additional num of worker for data loading
    cfg.num_workers = 0

    # Max threads used by PyTorch
    cfg.num_threads = 1

    # ----------------------------------------------------------------------- #
    # Debug
    # ----------------------------------------------------------------------- #

    cfg.tensorboard_each_run = True
    cfg.tensorboard_iter = False
    cfg.tensorboard_agg = True
    
    cfg.neptune_agg = False
    cfg.neptune_each_run = False
    
    cfg.draw_learned_graph = False
    
    # ----------------------------------------------------------------------- #
    # Dataset options
    # ----------------------------------------------------------------------- #
    cfg.dataset = CN()
    
    # Dir to load the dataset. If the dataset is downloaded, this is the
    # cache dir
    cfg.dataset.dir = ''
    # normalize data input
    cfg.dataset.normalize = True
    
    cfg.dataset.num_classes = 1
    cfg.dataset.train_file = 'train.csv'
    cfg.dataset.test_file = 'test.csv'
    cfg.dataset.stride = 1
    cfg.dataset.has_adj = False
    cfg.dataset.aug_ocvar = True
    cfg.dataset.aug_ocvar_on_node = False
    cfg.dataset.no_ocvar = False
    cfg.dataset.scaler_type = 'minmax'    
    cfg.dataset.use_indep_vars = False
    cfg.dataset.ocvar_dim = 2

    cfg.dataset.horizon = 1
    cfg.dataset.val_split = 0.2
    cfg.dataset.test_split = 0.2
    
    # Sequence length
    cfg.dataset.window_size = 15
    
    # ----------------------------------------------------------------------- #
    # Training options
    # ----------------------------------------------------------------------- #
    cfg.train = CN()

    # Total graph mini-batch size
    cfg.train.batch_size = 512

    # Save model checkpoint every checkpoint period epochs
    cfg.train.ckpt_period = 5
    cfg.train.draw_period = 5

    # Resume training from the latest checkpoint in the output directory
    cfg.train.auto_resume = False

    # The epoch to resume. -1 means resume the latest epoch.
    cfg.train.epoch_resume = -1

    # Clean checkpoint: only keep the last ckpt
    cfg.train.ckpt_clean = True
    
    # Gradient clipping
    cfg.train.clip_grad = False
    
    cfg.train.max_grad_norm = 1
    
    cfg.train.load_pretrained = False
    
    cfg.train.pretrained_model_path = ''
    
    # early stop patience till triggered
    cfg.train.early_stop_patience = 100   
    
    # early stop min steps 
    cfg.train.early_stop_min = 100
    
    # ----------------------------------------------------------------------- #
    # Optimizer options
    # ----------------------------------------------------------------------- #
    cfg.optim = CN()

    cfg.optim.criterion = 'rmse'

    # optimizer: sgd, adam
    cfg.optim.optimizer = 'adam'

    # Base learning rate
    cfg.optim.base_lr = 0.01

    # L2 regularization
    cfg.optim.weight_decay = 1e-5

    # SGD momentum
    cfg.optim.momentum = 0.9

    # scheduler: none, steps, plateau
    cfg.optim.scheduler = 'plateau'

    # Steps for 'steps' policy (in epochs)
    cfg.optim.steps = [30, 60, 90]

    # Learning rate multiplier for 'steps' policy
    cfg.optim.lr_decay = 0.1

    # Maximal number of epochs
    cfg.optim.max_epochs = 30
    
    # Learning rate multiplier for 'plateau' policy
    cfg.optim.factor = 0.95
    
    cfg.optim.patience = 2
    
    # Minimum learning rate
    cfg.optim.min_lr = 1e-4
    
    # ----------------------------------------------------------------------- #
    # Shared model paramters
    # ----------------------------------------------------------------------- #
    cfg.model = CN()
    
    cfg.model.type = ''    
    
    # ----------------------------------------------------------------------- #
    # Shared model paramters
    # ----------------------------------------------------------------------- #
    cfg.model.activation = 'relu'
    cfg.model.output_dim = 0
    cfg.model.do_norm = True    
    cfg.model.norm_func = 'batch'  
    cfg.model.dropout = 0.
    
    # ----------------------------------------------------------------------- #
    # Autoencoder model paramters
    # ----------------------------------------------------------------------- #
    cfg.model.ae = CN()
    cfg.model.ae.latent_dim = 8
    cfg.model.ae.hidden_dims = [20]
    
    # ----------------------------------------------------------------------- #
    # Feed forward networks paramters
    # ----------------------------------------------------------------------- #
    cfg.model.fnn = CN()
    cfg.model.fnn.hidden_dims = [20]
    
    # ----------------------------------------------------------------------- #
    # Autoencoder model paramters
    # ----------------------------------------------------------------------- #
    cfg.model.usad = CN()
    cfg.model.usad.latent_dim = 2
    cfg.model.usad.n_hidden_layers = 2
    cfg.model.usad.alpha = 0.5
    cfg.model.usad.final_act = 'sigmoid'
    cfg.model.usad.warmup_epochs = 30
    
    # ----------------------------------------------------------------------- #
    # RNN model paramters
    # ----------------------------------------------------------------------- #
    cfg.model.rnn = CN()
    cfg.model.rnn.rnn_type = 'gru'
    cfg.model.rnn.n_layers = 1
    cfg.model.rnn.hidden_dim = 50
    cfg.model.rnn.bidirectional = False
    
    # ----------------------------------------------------------------------- #
    # ENCDEC-AD model paramters
    # ----------------------------------------------------------------------- #
    cfg.model.lstmae = CN()
    cfg.model.lstmae.n_layers = 1
    cfg.model.lstmae.hidden_dim = 10
    
    # ----------------------------------------------------------------------- #
    # 1DCNN model paramters
    # ----------------------------------------------------------------------- #
    cfg.model.cnn = CN()
    cfg.model.cnn.n_channels = 10
    cfg.model.cnn.kernel_size = 5
    cfg.model.cnn.n_layers = 3
    cfg.model.cnn.hidden_dim = 50
    cfg.model.cnn.stride = 1
    
    # ----------------------------------------------------------------------- #
    # GDN paramters
    # ----------------------------------------------------------------------- #
    cfg.model.gdn = CN()
    cfg.model.gdn.topk = 20
    cfg.model.gdn.embed_dim = 128
    cfg.model.gdn.out_inter_dim = 256
    cfg.model.gdn.out_layer_num = 2
    cfg.model.gdn.outer_dropout = 0.2
    
    # ----------------------------------------------------------------------- #
    # Shared GNN paramters
    # ----------------------------------------------------------------------- #

    # ----------------------------------------------------------------------- #
    # MTADGAT paramters
    # ----------------------------------------------------------------------- #
    cfg.model.mtadgat = CN()
    
    cfg.model.mtadgat.encoder = CN()
    cfg.model.mtadgat.encoder.encode = True
    cfg.model.mtadgat.encoder.in_depth = False
    cfg.model.mtadgat.encoder.conv_pad = True
    cfg.model.mtadgat.encoder.padding = 0
    cfg.model.mtadgat.encoder.stride = 1
    cfg.model.mtadgat.encoder.dilation = 1
    cfg.model.mtadgat.encoder.kernel_size = 5
    
    cfg.model.mtadgat.use_gatv2 = True
    
    cfg.model.mtadgat.num_feat_layers = 1
    cfg.model.mtadgat.num_temp_layers = 1
    
    cfg.model.mtadgat.feat_gat_embed_dim = 100
    cfg.model.mtadgat.time_gat_embed_dim = 100
    
    cfg.model.mtadgat.encode_edge_attr = True
    cfg.model.mtadgat.edge_embed_dim = 3
    cfg.model.mtadgat.num_edge_types = 4
        
    cfg.model.mtadgat.gru_n_layers = 1
    cfg.model.mtadgat.gru_hid_dim = 100
    
    cfg.model.mtadgat.fc_n_layers = 3
    cfg.model.mtadgat.fc_hid_dim = 100
    
    cfg.model.mtadgat.recon_n_layers = 1
    cfg.model.mtadgat.recon_hid_dim = 100
    
    cfg.model.mtadgat.decoder_hidden = 100
    
    cfg.model.mtadgat.forecast = True
    cfg.model.mtadgat.recons = True

    # ----------------------------------------------------------------------- #
    # GRELEN paramters
    # ----------------------------------------------------------------------- #
    cfg.model.grelen = CN()
    cfg.model.grelen.prior = [0.91, 0.03, 0.03, 0.03]
    cfg.model.grelen.graph_learner_n_hid = 64
    cfg.model.grelen.graph_learner_n_head_dim = 32
    cfg.model.grelen.graph_learner_head = 4
    cfg.model.grelen.num_rnn_layers = 1
    cfg.model.grelen.gru_n_dim = 64
    cfg.model.grelen.temperature = 0.5
    cfg.model.grelen.max_diffusion_step = 2
    
    # ----------------------------------------------------------------------- #
    # Temporal Graph neural networks paramters
    # ----------------------------------------------------------------------- #
    cfg.model.dyedgegat = CN()
    
    cfg.model.dyedgegat.feat_input_node = 3
    cfg.model.dyedgegat.feat_target_node = 1
    cfg.model.dyedgegat.feat_input_edge = 1
    
    # node dynamics extraction
    cfg.model.dyedgegat.node_encoder_type = 'gru'
    cfg.model.dyedgegat.node_encoder_mode = 'simple'
    cfg.model.dyedgegat.learn_sys = False
    cfg.model.dyedgegat.contr_encoder_type = 'gru'
    cfg.model.dyedgegat.temp_node_embed_dim = 16
    
    # static edge construction (for the ablation study)
    cfg.model.dyedgegat.infer_graph = False
    cfg.model.dyedgegat.feat_edge_hid_dim = 128
    cfg.model.dyedgegat.topk = 20
    cfg.model.dyedgegat.add_self_loop = True
    
    # dynamic edge construction
    cfg.model.dyedgegat.infer_temporal_edge = True
    cfg.model.dyedgegat.aug_feat_edge_attr = True 
    cfg.model.dyedgegat.edge_aggr = 'dot'
    cfg.model.dyedgegat.temp_edge_embed_dim = 1 
    cfg.model.dyedgegat.temp_edge_hid_dim = 100
    cfg.model.dyedgegat.temporal_window = 5
    cfg.model.dyedgegat.temporal_kernel = 5
    cfg.model.dyedgegat.use_time_encoding = True
    cfg.model.dyedgegat.time_dim = 5
    
    # graph module
    cfg.model.dyedgegat.gnn_type = 'gin'
    cfg.model.dyedgegat.gnn_embed_dim = 20
    cfg.model.dyedgegat.num_gnn_layers = 1
    
    # normalization
    cfg.model.dyedgegat.do_encoder_norm = False
    cfg.model.dyedgegat.encoder_norm_type = 'batch'
    cfg.model.dyedgegat.do_gnn_norm = True
    cfg.model.dyedgegat.gnn_norm_type = 'batch'
    cfg.model.dyedgegat.do_decoder_norm = False
    cfg.model.dyedgegat.decoder_norm_type = 'batch'
    
    # reconstruction
    cfg.model.dyedgegat.recon_hidden_dim = 20
    cfg.model.dyedgegat.num_recon_layers = 1
    cfg.model.dyedgegat.flip_output = False
    
    # ----------------------------------------------------------------------- #
    # Tasks
    # ----------------------------------------------------------------------- #
    cfg.task = CN()
    
    # Select task type: 'anomaly', 'forecasting', 'mapping'
    cfg.task.type = 'anomaly'
    
    cfg.task.track_graph = True
    
    # Select task metric: 'f1', 'recall', 'precision', 'auc', 'all', 'last', 'first'
    cfg.task.metric = 'all'
    
    # Select task anomaly score function: 'base', 'diff', 'node_scaled', 'att_scaled', 'node_scaled_topk
    cfg.task.anomaly_score_func = 'node_scaled'
    
    cfg.task.anomaly_score_metric = 'mae'

    # Hyperparameter for point adjustment. If K = 0, it is equal to the F1PA and if K = 100, it is equal to the F1.
    cfg.task.anomaly_score_pak = 50
    
    # Select task anomaly score topk
    cfg.task.anomaly_score_topk = 1
    
    # Select task anomaly score threshold mode: 'best', 'val'
    cfg.task.anomaly_score_thresh_mode = 'val'

    # Set task anomaly score threshold % percentile of max eval error score
    cfg.task.anomaly_score_thresh = 1.0
    
    # Set num steps to consider fault as detected
    cfg.task.detection_delay_ts = 30
    
    # Set task anomaly score sliding window size for smoothing
    cfg.task.anomaly_score_sw = 3
    
    # Select task level: 'graph', 'node'
    cfg.task.level = 'graph'
    
    # Select task training type: 'forecast', 'reconst', 'combined', 'mapping'
    cfg.task.train_type = 'forecast' 

    cfg.task.combine_type = 'mapping'

    # factor for balancing the reconstruction loss
    cfg.task.gamma = 1.


def dump_cfg(cfg, cfg_file=None):
    r"""
    Dumps the config to the output directory specified in
    :obj:`cfg.out_dir`
    Args:
        cfg (CfgNode): Configuration node
    """
    if cfg_file is None:
        os.makedirs(cfg.out_dir, exist_ok=True)
        cfg_file = os.path.join(cfg.out_dir, cfg.cfg_dest)
    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)


def load_yaml(yaml_path):
    with open(yaml_path) as file:
        return yaml.safe_load(file)


def load_cfg(cfg_file):
    r"""
    Load configurations from file path
    Args:
        cfg_file (string): config file path
    """
    cfg.merge_from_file(cfg_file)
    return cfg


def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)
    

def makedirs_rm_exist(dir):
    dir = os.path.abspath(dir)
    if os.path.isdir(dir):
        shutil.rmtree(dir, onerror=remove_readonly)
    os.makedirs(dir, exist_ok=True)


def get_fname(fname):
    r"""
    Extract filename from file name path
    Args:
        fname (string): Filename for the yaml format configuration file
    """
    fname = fname.split('/')[-1]
    if fname.endswith('.yaml'):
        fname = fname[:-5]
    elif fname.endswith('.yml'):
        fname = fname[:-4]
    return fname


def get_out_dir(out_dir, fname):
    fname = get_fname(fname)
    cfg.fname = fname
    out_dir = f'run/{out_dir}/{cfg.task.type}/{cfg.case}/{fname}'
    return out_dir


def set_out_dir(out_dir, fname, exist_ok=True):
    r"""
    Create the directory for full experiment run
    Args:
        out_dir (string): Directory for output, specified in :obj:`cfg.out_dir`
        fname (string): Filename for the yaml format configuration file
    """
    cfg.out_dir = get_out_dir(out_dir, fname)
    # Make output directory
    if cfg.train.auto_resume or exist_ok:
        os.makedirs(cfg.out_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.out_dir)
    return cfg.out_dir


def set_run_dir(out_dir, seed=None):
    r"""
    Create the directory for each random seed experiment run
    Args:
        out_dir (string): Directory for output, specified in :obj:`cfg.out_dir`
        fname (string): Filename for the yaml format configuration file
    """
    if seed is None:
        seed = cfg.seed
    cfg.run_dir = f'{out_dir}/{seed}'
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)

set_cfg(cfg)