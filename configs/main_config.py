config = {
    'seed': 3407,
    'train': {
        'epochs'           : 1500,
        'print_per_epochs' : 1,
        'eval_per_epochs'  : 10,
        'sample_per_epochs': 100,
    },
    'model': {
        'channel1': 128,
        'channel2': 256,
        'conv': {
            'stride'     : 1,
            'dilation'   : 1,
            'init_scale' : 1.,
            'padding'    : 1,
            'kernel_size': 3
        },
        'groupnorm': {
            'num_groups': 32
        },
        'time_embedding': {
            'max_positions': 10000,
            'emb_dim1'     : 128,
            'emb_dim2'     : 512
        },
        'dropout': 0.1
    },
    'sde': {
        'beta_min' : 0.2,
        'beta_max' : 20.,
        'T'        : 1,
        'eps'      : 1e-5,
        'timesteps': 1000
    },
    'optim': {
        'lr'          : 2e-4,
        'beta1'       : 0.9,
        'beta2'       : 0.999,
        'eps'         : 1e-8,
        'weight_decay': 0,
        'warmup'      : 30,
        'grad_clip'   : 1.
    },
    'ema': {
        'decay': 0.9999
    },
    'sampling': {
        'n_img': 20
    }
}

def load_config():

    return config