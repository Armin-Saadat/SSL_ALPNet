import json

base = './exps/myexperiments_MIDDLE_0/mySSL_train_CHAOST2_Superpix_lbgroup0_scale_MIDDLE_vfold0_CHAOST2_Superpix_sets_0_1shot'
model_folder = 1
model_snapshot = 10000

conf = {
    'reload_base_path': base,
    'model_folder': model_folder,
    'model_snapshot': model_snapshot,
    'modelname': 'dlfcn_res101',
    'usealign': True,
    'optim_type': 'sgd',
    'num_workers': 4,
    'scan_per_load': -1,
    'label_sets': 0,
    'use_wce': True,
    'exp_prefix': 'test_vfold',
    'clsname': 'grid_proto',
    'n_steps': 4000,
    'exclude_cls_list': [2, 3],
    'eval_fold': 0,
    'dataset': 'CHAOST2_Superpix',
    'proto_grid_size': 8,
    'max_iters_per_load': 1000,
    'min_fg_data': 1,
    'seed': 1234,
    'save_snapshot_every': 25000,
    'superpix_scale': 'MIDDLE',
    'lr_step_gamma': 0.95,
    'support_idx': [4]
}

with open('test_config.json', 'w') as outfile:
    json.dump(conf, outfile)
