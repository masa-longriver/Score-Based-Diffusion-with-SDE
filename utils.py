import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import torchvision.utils as vutils

def make_dir(path):
    path_list = path.split('/')
    now_path = ""
    for i, dir in enumerate(path_list):
        if i == 0:
            continue
        else:
            now_path += f"/{dir}"
            if not os.path.exists(now_path):
                os.makedirs(now_path)

class Save():
    def __init__(self, config, dataset):
        now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        parent_path = os.path.join(os.getcwd(), 'log', dataset.lower(), now)
        self.model_path = os.path.join(parent_path, 'models')
        self.sample_path = os.path.join(parent_path, 'samples')
        self.tensor_path = os.path.join(parent_path, 'tensor')
        make_dir(self.model_path)
        make_dir(self.sample_path)
        make_dir(self.tensor_path)
   
    def save_model(self, epoch, model):
        file_nm = os.path.join(
            self.model_path, 
            f'epoch{epoch}_model_stete_dict.pt'
        )
        torch.save(model.state_dict(), file_nm)

    def save_img(self, epoch, sample):
        file_nm = os.path.join(self.sample_path, f'epoch{epoch}_sample.png')
        reverse_transform1 = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2)
        ])
        reverse_transform2 = transforms.Compose([
            transforms.Lambda(lambda x: x * 255.),
            transforms.Lambda(lambda x: x.cpu().to(dtype=torch.int8))
        ])
        sample = reverse_transform1(sample)
        self.save_tensor(epoch, sample)
        sample = reverse_transform2(sample)
        print(sample)
        grid = vutils.make_grid(sample, nrow=5, padding=2)
        plt.axis('off')
        plt.imshow(grid.permute(1, 2, 0))
        plt.savefig(file_nm, bbox_inches='tight')
        plt.close()
    
    def save_tensor(self, epoch, sample):
        file_nm = os.path.join(self.tensor_path, f'epoch{epoch}_tensor.pt')
        torch.save(sample, file_nm)