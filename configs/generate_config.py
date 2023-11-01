import os

config = {
    'model_path': os.path.join(
        os.getcwd(), 
        'log/cifar10/20231101_175740/models/epoch1_model_state_dict.pt'
    )
}

def load_config_generate():

    return config