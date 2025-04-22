import os
import random

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from src.data.graph_dataset import MaterialsGraphDataset
from src.utils.transforms import SO3RotateAndJitter

def split_dataset(dataset_len, train_ratio=0.8, val_ratio=0.1, seed=42):
    indices = list(range(dataset_len))
    random.seed(seed)
    random.shuffle(indices)

    train_end = int(train_ratio * dataset_len)
    val_end   = int((train_ratio + val_ratio) * dataset_len)

    return indices[:train_end], indices[train_end:val_end], indices[val_end:]

def get_loaders(
    data_dir,
    target=None,
    batch_size=32,
    num_workers=0,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    jitter_std=0.01
):
    full_dataset = MaterialsGraphDataset(data_dir, target=target)

    train_idx, val_idx, test_idx = split_dataset(len(full_dataset),
                                                 train_ratio, val_ratio, seed)
    train_dataset = MaterialsGraphDataset(data_dir, target=target)
    train_dataset.transform = SO3RotateAndJitter(jitter_std=jitter_std)
    train_dataset = Subset(train_dataset, train_idx)

    val_dataset   = Subset(full_dataset, val_idx)
    test_dataset  = Subset(full_dataset, test_idx)

    print(f"Total {len(full_dataset)} Data: Train({len(train_dataset)}) / Val({len(val_dataset)}) / Test({len(test_dataset)})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader