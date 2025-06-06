# Copyright Universitat Politècnica de Catalunya 2024 https://imatge.upc.edu
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import random
import math
import os.path as osp
import numpy as np
import pickle as pk

import torch
from torch_geometric.loader import DataLoader

from src.data.figshare_dataset import Figshare_Dataset

def get_loaders(
    data_dir,
    target='formation_energy_per_atom',
    batch_size=32,
    num_workers=0,
    radius=5.0,
    seed=42,
    dataset_name='mpjv',
    max_neighbors=25,
    norm=False,
    mean=0.0,
    std=1.0
):
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    
    # hard-coded
    if isinstance(target, list):
        if len(target) == 1:
            target = target[0]
        else:
            raise TypeError(f'Data Target Name must be 1')
    else:
        pass
        
    train_pth = osp.join(data_dir, "train.pkl")
    val_pth = osp.join(data_dir, "val.pkl")
    test_pth = osp.join(data_dir, "test.pkl")

    try:
        data_train = pk.load(open(train_pth, "rb"))
        data_val = pk.load(open(val_pth, "rb"))
        data_test = pk.load(open(test_pth, "rb"))
    except:
        raise Exception("Dataset not found. Run >> python data make_dataset.py")
    
    targets_train = []
    dat_train = []
    targets_val = []
    dat_val = []
    targets_test = []
    dat_test = []
    for split, datalist, targets in zip([data_train, data_val, data_test], 
                                    [dat_train, dat_val, dat_test],
                                    [targets_train, targets_val, targets_test]):
        for i in split:
            if (
                i[target] is not None
                and i[target] != "na"
                and not math.isnan(i[target])
            ):
                datalist.append(i)
                targets.append(i[target])

    prefix = dataset_name+"_"+str(radius)+"_"+str(max_neighbors)+"_"+target+"_"+str(seed)
    dataset_train = Figshare_Dataset(root=data_dir, data=dat_train, targets=targets_train, radius=radius, max_neigh=max_neighbors, name=prefix+"_train", mode="train", augment=True, mean=mean, std=std, norm=norm)
    dataset_val = Figshare_Dataset(root=data_dir, data=dat_val, targets=targets_val, radius=radius, max_neigh=max_neighbors, name=prefix+"_val", mode="val", mean=mean, std=std, norm=norm)
    dataset_test = Figshare_Dataset(root=data_dir, data=dat_test, targets=targets_test, radius=radius, max_neigh=max_neighbors, name=prefix+"_test", mode="test", mean=mean, std=std, norm=norm)
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, persistent_workers=True,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, persistent_workers=True,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, persistent_workers=False,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=True)
    
    return train_loader, val_loader, test_loader



def create_train_val_test(data, val_ratio=0.1, test_ratio=0.1, seed=123):
    ids = list(np.arange(len(data)))
    n = len(data)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    random.seed(seed)
    random.shuffle(ids)
    ids_train = ids[:n_train]
    ids_val = ids[-(n_val + n_test): -n_test]
    ids_test = ids[-n_test:]
    return ids_train, ids_val, ids_test