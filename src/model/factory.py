import logging


from ..config import cfg
from .dyedgegat import DyEdgeGAT
from .baseline.ae import Autoencoder
from .baseline.mlp import MLP
from .baseline.rnn import RNN
from .baseline.cnn import CNN
from .baseline.lstmae import LSTM_AE
from .baseline.usad import USAD
from .baseline.gdn import GDN
from .baseline.mtadgat import MTADGAT
from .baseline.grelen import GRELEN


BASELINE_MODELS = {
    'ae': Autoencoder,
    'cnn': CNN,
    'mlp': MLP,
    'fnn': MLP,
    'rnn': RNN,
    'lstmae': LSTM_AE,
    'usad': USAD,
    'gdn': GDN,
    'mtadgat': MTADGAT,
    'grelen': GRELEN,
    'dyedgegat': DyEdgeGAT,
}


def create_model(model_type=None, set_params=True):
    common_args = dict(
        dropout_prob=cfg.model.dropout,
        activation=cfg.model.activation,
        norm_func=cfg.model.norm_func if cfg.model.norm_func != 'none' else None,
        output_dim=cfg.model.output_dim if cfg.model.output_dim > 0 else None,
    ) # type: ignore
    
    if model_type == 'dyedgegat':    
        model = DyEdgeGAT(
            feat_input_node=cfg.model.dyedgegat.feat_input_node,
            feat_target_node=cfg.model.dyedgegat.feat_target_node,
            feat_input_edge=cfg.model.dyedgegat.feat_input_edge,
            node_encoder_type=cfg.model.dyedgegat.node_encoder_type,
            node_encoder_mode=cfg.model.dyedgegat.node_encoder_mode,
            topk=cfg.model.dyedgegat.topk,
            temp_node_embed_dim=cfg.model.dyedgegat.temp_node_embed_dim,
            temp_edge_embed_dim=cfg.model.dyedgegat.temp_edge_embed_dim,
            temp_edge_hid_dim=cfg.model.dyedgegat.temp_edge_hid_dim,
            feat_edge_hid_dim=cfg.model.dyedgegat.feat_edge_hid_dim,
            temporal_window=cfg.model.dyedgegat.temporal_window,
            temporal_kernel=cfg.model.dyedgegat.temporal_kernel,
            aug_feat_edge_attr=cfg.model.dyedgegat.aug_feat_edge_attr,
            gnn_type=cfg.model.dyedgegat.gnn_type,
            do_encoder_norm=cfg.model.dyedgegat.do_encoder_norm,
            do_gnn_norm=cfg.model.dyedgegat.do_gnn_norm,
            do_decoder_norm=cfg.model.dyedgegat.do_decoder_norm,
            encoder_norm_type=cfg.model.dyedgegat.encoder_norm_type,
            decoder_norm_type=cfg.model.dyedgegat.decoder_norm_type,
            gnn_norm_type=cfg.model.dyedgegat.gnn_norm_type,
            num_gnn_layers=cfg.model.dyedgegat.num_gnn_layers,
            gnn_embed_dim=cfg.model.dyedgegat.gnn_embed_dim, 
            infer_static_graph=cfg.model.dyedgegat.infer_graph,
            infer_temporal_edge=cfg.model.dyedgegat.infer_temporal_edge,
            recon_hidden_dim=cfg.model.dyedgegat.recon_hidden_dim,
            num_recon_layers=cfg.model.dyedgegat.num_recon_layers,
            use_time_encoding=cfg.model.dyedgegat.use_time_encoding,
            time_dim=cfg.model.dyedgegat.time_dim,
            dropout=cfg.model.dropout,
            edge_aggr=cfg.model.dyedgegat.edge_aggr,
            aug_control=cfg.dataset.aug_ocvar_on_node,
            flip_output=cfg.model.dyedgegat.flip_output,
            learn_sys=cfg.model.dyedgegat.learn_sys,
        )
    elif model_type == 'gdn':    
        model = BASELINE_MODELS[model_type](
            node_num=cfg.dataset.n_nodes,
            input_dim=cfg.dataset.window_size,
            topk=cfg.model.gdn.topk,
            dropout=cfg.model.dropout,
            outer_dropout=cfg.model.gdn.outer_dropout,
            out_layer_num=cfg.model.gdn.out_layer_num,
            out_layer_inter_dim=cfg.model.gdn.out_inter_dim,
            dim=cfg.model.gdn.embed_dim,
        )
    elif model_type == 'mtadgat':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes,
            window_size=cfg.dataset.window_size,
            encode=cfg.model.mtadgat.encoder.encode,
            kernel_size=cfg.model.mtadgat.encoder.kernel_size,
            use_gatv2=cfg.model.mtadgat.use_gatv2,
            feat_gat_embed_dim=cfg.model.mtadgat.feat_gat_embed_dim,
            time_gat_embed_dim=cfg.model.mtadgat.time_gat_embed_dim,
            gru_n_layers=cfg.model.mtadgat.gru_n_layers,
            gru_hid_dim=cfg.model.mtadgat.gru_hid_dim,
            forecast_n_layers=cfg.model.mtadgat.fc_n_layers,
            forecast_hid_dim=cfg.model.mtadgat.fc_hid_dim,
            recon_n_layers=cfg.model.mtadgat.recon_n_layers,
            recon_hid_dim=cfg.model.mtadgat.recon_hid_dim,
        )
    elif model_type == 'grelen':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes,
            window_size=cfg.dataset.window_size,
            graph_learner_n_hid=cfg.model.grelen.graph_learner_n_hid,
            graph_learner_n_head_dim=cfg.model.grelen.graph_learner_n_head_dim,
            graph_learner_head=cfg.model.grelen.graph_learner_head,
            num_rnn_layers=cfg.model.grelen.num_rnn_layers,
            gru_n_dim=cfg.model.grelen.gru_n_dim,
            temperature=cfg.model.grelen.temperature,
            max_diffusion_step=cfg.model.grelen.max_diffusion_step,
        )
    elif model_type == 'ae':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes,
            hidden_dims=cfg.model.ae.hidden_dims,
        )    
    elif model_type == 'cnn':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes,
            window_size=cfg.dataset.window_size,
            n_layers=cfg.model.cnn.n_layers,
            n_channels=cfg.model.cnn.n_channels,
            kernel_size=cfg.model.cnn.kernel_size,
            stride=cfg.model.cnn.stride,
            n_hidden=cfg.model.cnn.hidden_dim,
        )
    elif model_type == 'fnn' or model_type == 'mlp':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes,
            hidden_dims=cfg.model.fnn.hidden_dims,
        )
    elif model_type == 'rnn':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes,
            hidden_dim=cfg.model.rnn.hidden_dim,
            n_layers=cfg.model.rnn.n_layers,
            rnn_type=cfg.model.rnn.rnn_type,
            bidirectional=cfg.model.rnn.bidirectional,
        )
    elif model_type == 'lstmae':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes,
            hidden_dim=cfg.model.lstmae.hidden_dim,
        )
    elif model_type == 'usad':
        model = BASELINE_MODELS[model_type](
            **common_args,
            input_dim=cfg.dataset.n_nodes,
            window_size=cfg.dataset.window_size,
            latent_dim=cfg.model.usad.latent_dim,
            final_act=cfg.model.usad.final_act,
            n_hidden_layers=cfg.model.usad.n_hidden_layers,
        )
    else:
        raise ValueError(f'Model {model_type} not supported. Should be one of {BASELINE_MODELS.keys()}')
          
    model.to(cfg.device)
    if set_params:
        params_count = sum([p.numel() for p in model.parameters()])
        cfg.params = int(params_count)
        logging.info(f'Model {model.__class__.__name__} parameters: {params_count}, deployed to {cfg.device}')
        logging.info(model)
    return model

