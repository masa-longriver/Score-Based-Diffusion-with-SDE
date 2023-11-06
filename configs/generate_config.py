import os

def load_config_generate(dataset_nm):
    config = {
        'model_path': os.path.join(
            'log',
            dataset_nm,
            '20231101_175740/models/epoch2999_model_stete_dict.pt'
        )
    }
    
    return config