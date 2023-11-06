import argparse

import torch

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

    model = UNet(config).to(config['device'])
    state_dict = torch.load(generate_config['model_path'])
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('_orig_mod.', '').replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
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