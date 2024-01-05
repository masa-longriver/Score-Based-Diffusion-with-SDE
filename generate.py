import argparse

import torch
import torch.nn as nn

from configs.generate_config import load_config_generate
from configs.main_config import load_config
from configs.cifar10_config import load_config_cifar10
from configs.food_101_config import load_config_food_101
from configs.food_101_small_config import load_config_food_101_small
from model import UNet
from sampling import sampling
from sde import VPSDE
from utils import Save


parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    help="Select the dataset from ['food-101', 'food-101-small', 'cifar10']"
)
args = parser.parse_args()


if __name__ == '__main__':
    config = load_config()
    generate_config = load_config_generate(args.dataset.lower())
    if args.dataset.lower() == 'cifar10':
        config['data'] = load_config_cifar10()
    elif args.dataset.lower() == 'food-101':
        config['data'] = load_config_food_101()
    elif args.dataset.lower() == 'food-101-small':
        config['data'] = load_config_food_101_small()
    else:
        raise NotImplementedError("Dataset is not supported.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    model = torch.load(generate_config['model_path'])
    model = model.to(config['device']).to(dtype=torch.float32)
    model = nn.DataParallel(model)
    SDE = VPSDE(config)
    save = Save(args.dataset)

    shape = (
                config['sampling']['n_img'], config['data']['channel'],
                config['data']['height'], config['data']['width']
            )
    sample = sampling(config, shape, model, SDE)
    epoch = 'only_sampling'
    save.save_model(epoch, model)
    save.save_img(epoch, sample)