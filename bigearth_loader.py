import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_and_extract_archive, download_url
import torch
import torchvision
import tifffile
from torchvision import transforms
from pytorch_lightning import LightningDataModule

ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
RGB_BANDS = ['B04', 'B03', 'B02']

BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B11': 1079.19066363,
        'B12': 818.86747235
    }
}

LABELS = [
    'Agro-forestry areas', 'Airports',
    'Annual crops associated with permanent crops', 'Bare rock',
    'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas',
    'Coastal lagoons', 'Complex cultivation patterns', 'Coniferous forest',
    'Construction sites', 'Continuous urban fabric',
    'Discontinuous urban fabric', 'Dump sites', 'Estuaries',
    'Fruit trees and berry plantations', 'Green urban areas',
    'Industrial or commercial units', 'Inland marshes', 'Intertidal flats',
    'Land principally occupied by agriculture, with significant areas of '
    'natural vegetation', 'Mineral extraction sites', 'Mixed forest',
    'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
    'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land',
    'Port areas', 'Rice fields', 'Road and rail networks and associated land',
    'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
    'Sparsely vegetated areas', 'Sport and leisure facilities',
    'Transitional woodland/shrub', 'Vineyards', 'Water bodies', 'Water courses'
]

NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value)
    return img


class Bigearthnet(Dataset):
    url = 'https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz'
    # url = 'http://bigearth.net/downloads/BigEarthNet-v1.0.tar.gz'
    
    subdir = 'BigEarthNet-v1.0'
    list_file = {
        'train': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-train.txt',
        'val': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt',
        'test': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt'
    }
    bad_patches = [
        'http://bigearth.net/static/documents/patches_with_seasonal_snow.csv',
        'http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv'
    ]

    def __init__(self, root, split, bands=None, transform=None, target_transform=None, download=False, use_new_labels=True, channels= [0,1,2,3,4,5,6,7,8,9,10,11]):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else ALL_BANDS
        self.transform = transform
        self.target_transform = target_transform
        self.use_new_labels = use_new_labels
        self.resize = torchvision.transforms.Resize((120, 120), interpolation=Image.BICUBIC)
        self.c = channels

        # if download:
        #     download_and_extract_archive(self.url, self.root)
        #     download_url(self.list_file[self.split], self.root, f'{self.split}.txt')
        #     for url in self.bad_patches:
        #         download_url(url, self.root)

        bad_patches = set()
        for url in self.bad_patches:
            filename = Path(url).name
            with open(self.root / filename) as f:
                bad_patches.update(f.read().splitlines())
        
        # self.samples = []
        # with open(self.root / f'{self.split}.txt') as f:
        #     for patch_id in f.read().splitlines():
        #         self.samples.append(self.root / self.subdir / patch_id)

        self.samples = []
        with open(self.root / f'{self.split}.txt') as f:
            for patch_id in f.read().splitlines():
                if patch_id not in bad_patches:
                    self.samples.append(self.root / self.subdir / patch_id)
        print(split, len(self.samples))

    def __getitem__(self, index):
        path = self.samples[index]
        patch_id = path.name

        channels = []
        for b in self.bands:
            # ch = rasterio.open(path / f'{patch_id}_{b}.tif').read(1)
            img = tifffile.imread(path / f'{patch_id}_{b}.tif')
            ch = img.astype('float32')
            ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
            ch = self.resize(torch.tensor(ch).unsqueeze(0)).squeeze(0)
            channels.append(ch)
        img = torch.stack(channels, dim=0).float()
        img = img[self.c, :, :]

        with open(path / f'{patch_id}_labels_metadata.json', 'r') as f:
            labels = json.load(f)['labels']
        if self.use_new_labels:
            target = self.get_multihot_new(labels)
        else:
            target = self.get_multihot_old(labels)

        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_multihot_old(labels):
        target = np.zeros((len(LABELS),), dtype=np.float32)
        for label in labels:
            target[LABELS.index(label)] = 1
        return target

    @staticmethod
    def get_multihot_new(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
        for label in labels:
            if label in GROUP_LABELS:
                target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
            elif label not in set(NEW_LABELS):
                continue
            else:
                target[NEW_LABELS.index(label)] = 1
        return target
    
# def load_BigEarthNet(channels=[0,1,2,3,4,5,6,7,8,9,10,11], batch_size=16, seed = None):
#     path = "/projects/talwalkar/datasets/bigearthnet/"
#     train_transform = torchvision.transforms.Compose([
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.RandomVerticalFlip(),
#         torchvision.transforms.RandomRotation(90),
#         torchvision.transforms.Resize((224, 224)),
#         #torchvision.transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
#     ])
#     val_transform = torchvision.transforms.Compose([
#         torchvision.transforms.Resize((224, 224)),
#     ])
#     train_dataset = Bigearthnet(root=path, split='train', transform=train_transform, channels = channels)
#     val_dataset = Bigearthnet(root=path, split='val', transform=val_transform, channels = channels)
#     train_dataset = random_subset(train_dataset, 0.1, seed = seed)
#     print("train:", len(train_dataset))
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

#     return train_loader, val_loader


class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def random_subset(dataset, frac, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(len(dataset)), int(frac * len(dataset)))
    return Subset(dataset, indices)

class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class BigearthnetDataModule(LightningDataModule):

    def __init__(self, bands=None, train_frac=None, val_frac=None, batch_size=32, num_workers=3, seed=42, channels=[0,1,2,3,4,5,6,7,8,9,10,11]):
        super().__init__()
        self.data_dir = "/projects/talwalkar/datasets/bigearthnet/"
        self.bands = bands
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.channels = channels

        self.train_dataset = None
        self.val_dataset = None

    @property
    def num_classes(self):
        return 19

    def setup(self, stage=None):
        train_transforms = self.train_transform()
        self.train_dataset = Bigearthnet(
            root=self.data_dir,
            split='train',
            bands=self.bands,
            transform=None,
            channels=self.channels
        )
        if self.train_frac is not None and self.train_frac < 1:
            self.train_dataset = random_subset(self.train_dataset, self.train_frac, self.seed)

        val_transforms = self.val_transform()
        
        self.val_dataset = Bigearthnet(
            root=self.data_dir,
            split='val',
            bands=self.bands,
            transform=None,
            channels=self.channels
        )
        if self.val_frac is not None and self.val_frac < 1:
            self.val_dataset = random_subset(self.val_dataset, self.val_frac, self.seed)
        print("training subset:", len(self.train_dataset))

    @staticmethod
    def train_transform():
        return transforms.Compose([
            transforms.Resize((96, 96), interpolation=Image.BICUBIC),
            # transforms.ToTensor()
        ])

    @staticmethod
    def val_transform():
        return transforms.Compose([
            transforms.Resize((96, 96), interpolation=Image.BICUBIC),
            # transforms.ToTensor()
        ])

    def train_dataloader(self):
        # return InfiniteDataLoader(
        #     dataset=self.train_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=self.num_workers,
        #     pin_memory=True,
        #     drop_last=True
        # )
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        # return InfiniteDataLoader(
        #     dataset=self.val_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     pin_memory=True,
        #     drop_last=True
        # )
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

def load_BigEarthNet(channels=[1,2,3,4,5,6,7,8,9,10,11,12], batch_size=16):
    datamodule = BigearthnetDataModule(
        batch_size=batch_size,
        train_frac=0.1,
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    return train_loader, val_loader