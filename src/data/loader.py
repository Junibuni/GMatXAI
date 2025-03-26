import os
import random

from torch_geometric.loader import DataLoader
from src.data.graph_dataset import MaterialsGraphDataset

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    total_size = len(dataset)
    indices = list(range(total_size))
    random.seed(seed)
    random.shuffle(indices)

    train_end = int(train_ratio * total_size)
    val_end = int((train_ratio + val_ratio) * total_size)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return train_indices, val_indices, test_indices


def get_loaders(
    data_dir,
    target="all",
    batch_size=32,
    num_workers=0,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
):
    dataset = MaterialsGraphDataset(data_dir, target=target)
    train_idx, val_idx, test_idx = split_dataset(dataset, train_ratio, val_ratio, seed)

    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
