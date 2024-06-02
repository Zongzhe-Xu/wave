import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
import copy
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling
from torchmetrics.classification import MultilabelAccuracy
import wandb
import copy
from functools import partial
from pathlib import Path
import tifffile
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets as datasets


# class SentinelNormalize:
#     """
#     Normalization for Sentinel-2 imagery, inspired from
#     https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
#     """
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, x, *args, **kwargs):
#         min_value = self.mean - 2 * self.std
#         max_value = self.mean + 2 * self.std
#         img = (x - min_value) / (max_value - min_value)
#         return img

class channelNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, *args, **kwargs):
        x = (x - self.mean) / self.std
        return x
    
def split_dataset(train_dataset, valid_split):
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_split * num_train)) if valid_split <= 1 else num_train - valid_split

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler
    
def load_cifar(root, num_classes, batch_size, permute=False, seed=1111, valid_split=-1, maxsize=None):
    if num_classes == 10:
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    else:
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

    train_transforms = [transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize] #transforms.Resize(224),
    val_transforms = [transforms.ToTensor(), normalize]

    cifar = datasets.CIFAR100 if num_classes == 100 else datasets.CIFAR10

    train_dataset = cifar(root=root, train=True, transform=transforms.Compose(train_transforms), download=True)
    test_dataset = cifar(root=root, train=False, transform=transforms.Compose(val_transforms))

    if valid_split > 0:
        valid_dataset = cifar(root=root, train=True, transform=transforms.Compose(val_transforms))
        train_sampler, valid_sampler = split_dataset(train_dataset, len(test_dataset) / len(train_dataset))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4, pin_memory=True)
    elif maxsize is not None:
        valid_dataset = cifar(root=root, train=True, transform=transforms.Compose(val_transforms))
        train_sampler, valid_sampler = split_dataset(train_dataset, maxsize)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if valid_split > 0:
        return train_loader, val_loader, test_loader
    return train_loader, test_loader


def load_bigearthnet(channels=[0,1,2,3,4,5,6,7,8,9,10,11], batch_size=16):
    mean_std =  torch.load("/home/zongzhex/gene-orca/datasets/geobench/bigearthnet/mean_std.pt")
    mean = mean_std['mean']
    std = mean_std['std']
    train_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(90),
        # torchvision.transforms.Resize((224, 224)),
        #torchvision.transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    ])
    test_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        # torchvision.transforms.Resize((224, 224)),
    ])
    train_dataset = bigearthnetDataset('train', channels=channels, transform=train_transform)
    test_dataset = bigearthnetDataset('test', channels=channels, transform=test_transform)
    print("Size of the train/val dataset:", len(train_dataset), len(test_dataset))
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=3, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=3, drop_last=False)
    return train_loader, test_loader


class bigearthnetDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, channels=[3,2,1]):
        self.root = Path(f"/home/zongzhex/gene-orca/datasets/geobench/bigearthnet/{split}")
        self.channels = channels
        self.split = split
        self.transform = transform
        #iterate through the folder and get the filenames
        filenames = []
        for filename in os.listdir(self.root):
            if filename.endswith(".pt"):
                filenames.append(filename)
        if split == 'test':
            new_files = []
            newroot = Path(f"/home/zongzhex/gene-orca/datasets/geobench/bigearthnet/val")
            for filename in os.listdir(newroot):
                if filename.endswith(".pt"):
                    new_files.append(filename)

        self.samples = []
        for fn in filenames:
            self.samples.append(self.root / fn)
        if split == 'test':
            for fn in new_files:
                self.samples.append(newroot / fn)

    def __getitem__(self, index):
        path = self.samples[index]
        data = torch.load(path)
        x = data['x']
        y = torch.tensor(data['y']).float()
        
        if self.transform is not None:
            x = self.transform(x)

        x = x[self.channels,:,:]

        return x, y

    def __len__(self):
        return len(self.samples)

class brickKilnDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, channels=[0,1,2,3,4,5,6,7,8,9,10,11,12]):
        self.root = Path(f"/home/zongzhex/gene-orca/datasets/geobench/brickkiln/{split}")
        self.channels = channels
        self.split = split
        self.transform = transform
        #iterate through the folder and get the filenames
        filenames = []
        for filename in os.listdir(self.root):
            if filename.endswith(".pt"):
                filenames.append(filename)
        if split == 'test':
            new_files = []
            newroot = Path(f"/home/zongzhex/gene-orca/datasets/geobench/brickkiln/val")
            for filename in os.listdir(newroot):
                if filename.endswith(".pt"):
                    new_files.append(filename)

        self.samples = []
        for fn in filenames:
            self.samples.append(self.root / fn)
        if split == 'test':
            for fn in new_files:
                self.samples.append(newroot / fn)

    def __getitem__(self, index):
        path = self.samples[index]
        data = torch.load(path)
        x = data['x']
        y = data['y']
        
        if self.transform is not None:
            x = self.transform(x)
        
        x = x[self.channels,:,:].float()

        return x, y

    def __len__(self):
        return len(self.samples)
    
def load_brick_kiln(channels=[0,1,2,3,4,5,6,7,8,9,10,11,12], batch_size=16):
    mean_std =  torch.load("/home/zongzhex/gene-orca/datasets/geobench/brickkiln/mean_std.pt")
    mean = mean_std['mean']
    std = mean_std['std']
    train_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(90),
        # torchvision.transforms.Resize((224, 224)),
        #torchvision.transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    ])
    test_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        # torchvision.transforms.Resize((224, 224)),
    ])
    train_dataset = brickKilnDataset('train', channels=channels, transform=train_transform)
    test_dataset = brickKilnDataset('test', channels=channels, transform=test_transform)
    print(len(train_dataset), len(test_dataset))
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=3, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=3, drop_last=False)
    return train_loader, test_loader


class EurosatDataset(torch.utils.data.Dataset):

    def __init__(self, split, transform=None, root = "/home/zongzhex/SatMAE/datasets/EuroSAT_MS/", channels=[0,1,2,3,4,5,6,7,8,9,11,12]):
        self.channel_mean = torch.tensor([[[[1354.4065]],[[1118.2460]],[[1042.9271]],[[947.6271]],[[1199.4736]],[[1999.7968]],[[2369.2229]],[[2296.8181]],[[732.0835]],[[12.1133]],[[1819.0067]],[[1118.9218]],[[2594.1438]]]])
        self.channel_std = torch.tensor([[[[245.7176]],[[333.0078]],[[395.0925]],[[593.7505]],[[566.4170]],[[861.1840]],[[1086.6313]],[[1117.9817]],[[404.9198]],[[4.7758]],[[1002.5877]],[[761.3032]],[[1231.5858]]]])
        # print(self.channel_mean.shape)
        # print(self.channel_std.shape)
        self.root = Path(root)
        self.channels = channels
        self.split = split
        self.transform = transform
        if split == 'train':
            with open(self.root / 'eurosat-train.txt') as f:
                filenames = f.read().splitlines()
        else:
            with open(self.root / 'eurosat-test.txt') as f:
                filenames = f.read().splitlines()
            with open(self.root / 'eurosat-val.txt') as f:
                filenames += f.read().splitlines()

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for fn in filenames:
            #change the format from .jpg to .tif
            fn = fn.replace('.jpg', '.tif')
            cls_name = fn.split('_')[0]
            self.samples.append(self.root / cls_name / fn)

    def __getitem__(self, index):
        path = self.samples[index]
        img = tifffile.imread(path)
        img = img.astype('float32')
        img = torch.tensor(img).permute(2, 0, 1)
        # img = img.unsqueeze(0)
        # img = normalize(img, use_8_bit=True)
        # img = img.squeeze(0)
        # img = img.float()/255
        img = img.unsqueeze(0)
        img = (img - self.channel_mean) / self.channel_std
        img = img.squeeze(0).float()
        img = img[self.channels]
        
        # img = Image.open(path)
        target = self.class_to_idx[path.parts[-2]]
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)

def load_EuroSAT(channels=[0,1,2,3,4,5,6,7,8,9,10,11,12], batch_size=16):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(90),
        # torchvision.transforms.Resize((224, 224)),
        #torchvision.transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    ])
    # test_transform = torchvision.transforms.Compose([
    #     # torchvision.transforms.Resize((224, 224)),
    # ])
    train_dataset = EurosatDataset('train', channels=channels, transform=train_transform)
    test_dataset = EurosatDataset('test', channels=channels)
    print(len(train_dataset), len(test_dataset))
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=3, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=3, drop_last=False)
    return train_loader, test_loader

class so2satDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, channels=None):
        self.root = Path(f"/home/zongzhex/gene-orca/datasets/geobench/so2sat/{split}")
        self.channels = channels
        self.split = split
        self.transform = transform
        #iterate through the folder and get the filenames
        filenames = []
        for filename in os.listdir(self.root):
            if filename.endswith(".pt"):
                filenames.append(filename)
        if split == 'test':
            new_files = []
            newroot = Path(f"/home/zongzhex/gene-orca/datasets/geobench/so2sat/val")
            for filename in os.listdir(newroot):
                if filename.endswith(".pt"):
                    new_files.append(filename)

        self.samples = []
        for fn in filenames:
            self.samples.append(self.root / fn)
        if split == 'test':
            for fn in new_files:
                self.samples.append(newroot / fn)

    def __getitem__(self, index):
        path = self.samples[index]
        data = torch.load(path)
        x = data['x']
        y = data['y']
        
        if self.transform is not None:
            x = self.transform(x)
        
        x = torch.cat((x[:8,:,:], torch.zeros(1, 224, 224), x[8:16,:,:], torch.zeros(2, 224, 224), x[16:,:,:]), 0)
        x = x[self.channels,:,:].float() # 10*120*120
        # add 0 filled channels to 0th and 7th index
        

        return x, y

    def __len__(self):
        return len(self.samples)
    
def load_so2sat(channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], batch_size=16): #32*32*18
    mean_std =  torch.load("/home/zongzhex/gene-orca/datasets/geobench/so2sat/mean_std.pt")
    mean = mean_std['mean']
    std = mean_std['std']
    train_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(90),
        # torchvision.transforms.Resize((224, 224)),
        #torchvision.transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    ])
    test_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        # torchvision.transforms.Resize((224, 224)),
    ])
    train_dataset = so2satDataset('train', channels=channels, transform=train_transform)
    test_dataset = so2satDataset('test', channels=channels, transform=test_transform)
    print(len(train_dataset), len(test_dataset))
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=3, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=3, drop_last=False)
    return train_loader, test_loader


class forestnetDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, channels=None):
        self.root = Path(f"/home/zongzhex/gene-orca/datasets/geobench/forestnet/{split}")
        self.channels = channels
        self.split = split
        self.transform = transform
        #iterate through the folder and get the filenames
        filenames = []
        for filename in os.listdir(self.root):
            if filename.endswith(".pt"):
                filenames.append(filename)
        if split == 'test':
            new_files = []
            newroot = Path(f"/home/zongzhex/gene-orca/datasets/geobench/forestnet/val")
            for filename in os.listdir(newroot):
                if filename.endswith(".pt"):
                    new_files.append(filename)

        self.samples = []
        for fn in filenames:
            self.samples.append(self.root / fn)
        if split == 'test':
            for fn in new_files:
                self.samples.append(newroot / fn)

    def __getitem__(self, index):
        path = self.samples[index]
        data = torch.load(path)
        x = data['x']
        y = data['y']
        
        if self.transform is not None:
            x = self.transform(x)
        
        x = x[self.channels,:,:].float()

        return x, y

    def __len__(self):
        return len(self.samples)
    
def load_forestnet(channels=[0,1,2,3,4,5], batch_size=16): #32*32*18
    mean_std =  torch.load("/home/zongzhex/gene-orca/datasets/geobench/forestnet/mean_std.pt")
    mean = mean_std['mean']
    std = mean_std['std']
    train_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(90),
        # torchvision.transforms.Resize((224, 224)),
        #torchvision.transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    ])
    test_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        # torchvision.transforms.Resize((224, 224)),
    ])
    train_dataset = forestnetDataset('train', channels=channels, transform=train_transform)
    test_dataset = forestnetDataset('test', channels=channels, transform=test_transform)
    print(len(train_dataset), len(test_dataset))
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=3, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=3, drop_last=False)
    return train_loader, test_loader


class pv4gerDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, channels=None):
        self.root = Path(f"/home/zongzhex/gene-orca/datasets/geobench/pv4ger/{split}")
        self.channels = channels
        self.split = split
        self.transform = transform
        #iterate through the folder and get the filenames
        filenames = []
        for filename in os.listdir(self.root):
            if filename.endswith(".pt"):
                filenames.append(filename)
        if split == 'test':
            new_files = []
            newroot = Path(f"/home/zongzhex/gene-orca/datasets/geobench/pv4ger/val")
            for filename in os.listdir(newroot):
                if filename.endswith(".pt"):
                    new_files.append(filename)

        self.samples = []
        for fn in filenames:
            self.samples.append(self.root / fn)
        if split == 'test':
            for fn in new_files:
                self.samples.append(newroot / fn)

    def __getitem__(self, index):
        path = self.samples[index]
        data = torch.load(path)
        x = data['x']
        y = data['y']
        
        if self.transform is not None:
            x = self.transform(x)
        
        x = x[self.channels,:,:].float()

        return x, y

    def __len__(self):
        return len(self.samples)
    

def load_pv4ger(channels=[0,1,2], batch_size=16): #32*32*18
    mean_std =  torch.load("/home/zongzhex/gene-orca/datasets/geobench/pv4ger/mean_std.pt")
    mean = mean_std['mean']
    std = mean_std['std']
    train_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(90),
        # torchvision.transforms.Resize((224, 224)),
        #torchvision.transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    ])
    test_transform = torchvision.transforms.Compose([
        channelNormalize(mean, std),
        # torchvision.transforms.Resize((224, 224)),
    ])
    train_dataset = pv4gerDataset('train', channels=channels, transform=train_transform)
    test_dataset = pv4gerDataset('test', channels=channels, transform=test_transform)
    print(len(train_dataset), len(test_dataset))
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler = sampler_train, batch_size=batch_size, num_workers=3, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler_val, num_workers=3, drop_last=False)
    return train_loader, test_loader