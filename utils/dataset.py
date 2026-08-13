import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from pathlib import Path


class SR_CACO2_Dataset(Dataset):
    def __init__(self, dataframe, base_dir, scale=2, cell_ids=None, tiles_id=None, transform=None, preprocessors=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing columns like '1024', '512', etc.
            base_dir (str): Path to folder containing hr_div_1, hr_div_2, etc.
            scale (int): Downscale factor (e.g., 2, 4, 8)
            transform (callable): Transform to apply to both HR and LR images
            cell_ids (list[int]): Which cell IDs to include (e.g., [0, 1])
        """
        self.df = dataframe.copy()

        # Filter by cell ID if given
        if cell_ids is not None:
            self.df = self.df[self.df['cell'].isin(cell_ids)]
        # Filter by tiles ID if given
        if tiles_id is not None:
            self.df = self.df[self.df['tile'].isin(tiles_id)]

        self.base_dir = base_dir
        self.scale = scale
        # Mapping scale to resolution column
        self.lr_div_folder = f"hr_div_{scale}"
        self.hr_div_folder = "hr_div_1"

        self.transform = transform
        self.preprocessors = preprocessors or []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lr_filename = row[str(1024 // self.scale)]  # e.g., "512"
        hr_filename = row["1024"]

        lr_path = os.path.join(self.base_dir, self.lr_div_folder, lr_filename)
        hr_path = os.path.join(self.base_dir, self.hr_div_folder, hr_filename)

        hr_image = Image.open(hr_path)
        lr_image = Image.open(lr_path)

        lr_np = np.array(lr_image) / 255.0
        for fn in self.preprocessors:
            lr_np = fn(lr_np)
        lr_np = np.clip(lr_np * 255.0, 0, 255).astype(np.uint8)
        lr_image = Image.fromarray(lr_np)

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return {'lr': lr_image, 'hr': hr_image}

# todo: FIX BIOSR DATASET


class BioSR_Dataset(Dataset):
    def __init__(self, base_dir, metadata, transform=None):
        self.base_dir = base_dir
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        lr_rel_path = entry[3]
        hr_rel_path = entry[4]

        lr_path = os.path.join(self.base_dir, lr_rel_path)
        hr_path = os.path.join(self.base_dir, hr_rel_path)

        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return {'lr': lr_image, 'hr': hr_image, 'scale': entry[5],
                'modality': entry[0],
                'cell': entry[1],
                'z': entry[2]
                }


# def get_dataloader(lr_dir, hr_dir, batch_size=16, shuffle=True, num_workers=4):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     dataset = SR_CACO2_Dataset(lr_dir, hr_dir, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=batch_size,
#                             shuffle=shuffle, num_workers=num_workers)
#     return dataloader

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_dataloader(dataset_type,
                   scale=2,
                   dataframe=None,
                   base_dir=None,
                   metadata=None,
                   cell_ids=None,
                   tiles_id=None,
                   batch_size=16,
                   shuffle=True,
                   num_workers=4,
                   subset_size=None,
                   preprocessors=None,
                   seed=42):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset_type == 'caco2':
        if dataframe is None:
            raise ValueError("DataFrame must be provided for CACO2 dataset.")
        dataset = SR_CACO2_Dataset(
            dataframe, base_dir=base_dir, scale=scale, transform=transform, cell_ids=cell_ids, tiles_id=tiles_id, preprocessors=preprocessors)

    elif dataset_type == 'biosr':
        dataset = BioSR_Dataset(base_dir, metadata, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    if subset_size is not None:
        random.seed(seed)
        indices = random.sample(range(len(dataset)),
                                k=min(subset_size, len(dataset)))
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader
