import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from configs.cifar10_config import load_config_cifar10
from configs.food_101_config import load_config_food_101
from configs.food_101_small_config import load_config_food_101_small

class Dataset():
    def __init__(self, dataset):
        self.dataset = dataset
        self.check_dataset()
        self.config = self.load_config()
    
    def check_dataset(self):
        supported_dataset = ['food-101', 'food-101-small', 'cifar10']
        if self.dataset.lower() not in supported_dataset:
            raise NotImplementedError("Dataset is not supported.")
        print(f"Dataset: {self.dataset}", flush=True)
    
    def load_config(self):
        if self.dataset.lower() == 'food-101':
            config = load_config_food_101()
        elif self.dataset.lower() == 'food-101-small':
            config = load_config_food_101_small()
        elif self.dataset.lower() == 'cifar10':
            config = load_config_cifar10()
        else:
            raise NotImplementedError("Dataset is not supported.")
        
        return config
    
    def create_dataset(self):
        transform = transforms.Compose([
            transforms.Resize((self.config['height'], self.config['width'])),
            transforms.RandomHorizontalFlip(
                p=self.config['horizontal_flip_rate']
            ),
            transforms.RandomGrayscale(p=self.config['grayscale_rate']),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)
        ])
        if self.dataset.lower() in ['food-101', 'food-101-small']:
            data_dir = self.config['path']
            ds = datasets.ImageFolder(root=data_dir, transform=transform)
            train_size = int(len(ds) * self.config['train_size'])
            eval_size  = len(ds) - train_size
            train_ds, eval_ds = random_split(
                ds, [train_size, eval_size]
            )
        elif self.dataset.lower() == 'cifar10':
            train_ds = datasets.CIFAR10(
                './cifar-10',
                train=True,
                download=True,
                transform=transform
            )
            eval_ds = datasets.CIFAR10(
                './cifar-10',
                train=False,
                download=True,
                transform=transform
            )

        return train_ds, eval_ds
    
    def get_dataloader(self):
        train_ds, eval_ds = self.create_dataset()
        train_dl = DataLoader(
            train_ds,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=2
        )
        eval_dl = DataLoader(
            eval_ds,
            batch_size=self.config['batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=2
        )
        
        return train_dl, eval_dl