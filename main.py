import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.main_config import load_config
from data import Dataset
from ema import EMA
from model import UNet
from run import run
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    torch.backends.cudnn.bechmark = True

    Dataset = Dataset(args.dataset)
    train_dl, eval_dl = Dataset.get_dataloader()
    config['data'] = Dataset.config

    SDE = VPSDE(config)
    model = UNet(config).to(config['device'])
    model = model.to(dtype=torch.float32)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['optim']['lr'],
        betas=(config['optim']['beta1'], config['optim']['beta2']),
        eps=config['optim']['eps'],
        weight_decay=config['optim']['weight_decay']
    )
    ema = EMA(config, model)
    save = Save(args.dataset)

    losses = []
    for epoch in range(config['train']['epochs']):
        train_loss = run(config, epoch, model, SDE, train_dl, optimizer, ema)
        losses.append(train_loss)
        str_log = f"Epoch: {epoch:>4}, train_loss: {train_loss:.5e}"
        if (epoch + 1) % config['train']['eval_per_epochs'] == 0:
            eval_loss = run(
                config, epoch, model, SDE, eval_dl, optimizer, ema, state='eval'
            )
            str_log += f", eval_loss: {eval_loss:.5e}"
        if (epoch + 1) % config['train']['print_per_epochs'] == 0 or epoch == 0:
            print(str_log, flush=True)

        if (epoch + 1) % config['train']['sample_per_epochs'] == 0:
            shape = (
                config['sampling']['n_img'], config['data']['channel'],
                config['data']['height'], config['data']['width']
            )
            sample = sampling(config, shape, model, SDE)
            save.save_img(epoch, sample)
        
        if (epoch + 1) % config['train']['save_per_epochs'] == 0:
            save.save_model(epoch, model)
            