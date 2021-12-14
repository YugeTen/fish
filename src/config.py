
dataset_defaults = {
    'fmow': {
        'epochs': 12,
        'batch_size': 64,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-4,
            'weight_decay': 0,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 24000,
        'meta_lr': 0.01,
        'meta_steps': 5,
        'selection_metric': 'acc_worst_region',
        'reload_inner_optim': True,
        'print_iters': 500
    },
    'camelyon': {
        'epochs': 20,
        'batch_size': 32,
        'optimiser': 'SGD',
        'optimiser_args': {
            'momentum': 0.9,
            'lr': 1e-4,
            'weight_decay': 0,
        },
        'pretrain_iters': 10000,
        'meta_lr': 0.01,
        'meta_steps': 3,
        'selection_metric': 'acc_avg',
        'reload_inner_optim': True,
        'print_iters': -1
    },
    'poverty': {
        'epochs': 200,
        'batch_size': 64,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-3,
            'weight_decay': 0,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 0,
        'meta_lr': 0.1,
        'meta_steps': 5,
        'selection_metric': 'r_wg',
        'reload_inner_optim': True,
        'print_iters': -1
    },
    'iwildcam': {
        'epochs': 18,
        'batch_size': 16,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-4,
            'weight_decay': 0.0,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 24000,
        'meta_lr': 0.01,
        'meta_steps': 10,
        'selection_metric': 'F1-macro_all',
        'reload_inner_optim': True,
        'print_iters': 500
    },
    'amazon': {
        'epochs': 3,
        'batch_size': 8,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 2e-6,
            'weight_decay': 0,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 31000,
        'meta_lr': 0.01,
        'meta_steps': 5,
        'selection_metric': '10th_percentile_acc',
        'reload_inner_optim': True,
        'print_iters': 500
    },
    'civil': {
        'epochs': 5,
        'batch_size': 16,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-5,
            'weight_decay': 0,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 20000,
        'meta_lr': 0.05,
        'meta_steps': 5,
        'selection_metric': 'acc_wg',
        'reload_inner_optim': True,
        'print_iters': 500
},
    'cdsprites': {
        'epochs': 100,
        'batch_size': 64,
        'optimiser': 'Adam',
        'optimiser_args': {
            'lr': 1e-3,
            'weight_decay': 0,
            'amsgrad': True,
            'betas': (0.9, 0.999),
        },
        'pretrain_iters': 0,
        'meta_lr': 0.15,
        'meta_steps': 15,
        'num_domains': 15,
        'selection_metric': 'acc_avg',
        'reload_inner_optim': False,
        'print_iters': -1
    },
}
