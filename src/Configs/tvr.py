import os
import yaml


cfg = {}


cfg['model_name'] = 'H_base'
cfg['root'] = '/project/prvr/dataset/'
cfg['data_root'] = '/project/prvr/dataset/'
cfg['dataset_name'] = 'tvr'
cfg['seed'] = 9527


cfg['visual_feature'] = 'clip'
cfg['collection'] = 'tvr'
cfg['map_size'] = 128
cfg['clip_scale_w'] = 0.4
cfg['frame_scale_w'] = 0.6
cfg['sim_thr'] = 0.5
cfg['model_root'] = os.path.join(cfg['root'], cfg['dataset_name'], 'results_ms', cfg['model_name'])
cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')
cfg['tb_dir'] = os.path.join(cfg['root'], cfg['dataset_name'], 'results_ms', cfg['model_name'])

# extra
cfg['sft_factor'] = 0.09


# dataset
cfg['num_workers'] = 8
cfg['no_core_driver'] = False
cfg['no_pin_memory'] = False
cfg['batchsize'] = 128


# opt
cfg['lr'] = 0.0003
cfg['lr_warmup_proportion'] = 0.01
cfg['wd'] = 0.01
cfg['margin'] = 0.1


# train
cfg['n_epoch'] = 100
cfg['max_es_cnt'] = 10
cfg['hard_negative_start_epoch'] = 20
cfg['hard_pool_size'] = 20
cfg['use_hard_negative'] = False
cfg['loss_factor'] = [0.05, 0.04, 8e-5, 0.09]
cfg['neg_factor'] = [0.15, 32, 1]
cfg['vl_coef'] = 0.1

# eval
cfg['eval_query_bsz'] = 50
cfg['eval_context_bsz'] = 100


# model
cfg['max_desc_l'] = 77
cfg['max_ctx_l'] = 128
cfg['sub_feat_size'] = 768
cfg['visual_feat_dim'] = 768
cfg['q_feat_size'] = 768
cfg['max_position_embeddings'] = 300
cfg['hidden_size'] = 384
cfg['n_heads'] = 4
cfg['input_drop'] = 0.2
cfg['drop'] = 0.2
cfg['initializer_range'] = 0.02


cfg['num_workers'] = 1 if cfg['no_core_driver'] else cfg['num_workers']
cfg['pin_memory'] = not cfg['no_pin_memory']


def get_cfg_defaults():
    return cfg