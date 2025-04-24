import os
import re
import json
import functools

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from src.utils.atom_info import ATOM_TYPES

def extract_atom_specie(formula, include_charge=False):
    match = re.match(r'^([A-Z][a-z]?)(\d*)([+-])?', formula)
    if not match:
        return None

    specie = match.group(1)
    charge_num = match.group(2)
    sign = match.group(3)

    if include_charge:
        if charge_num and sign:
            charge = int(charge_num) * (1 if sign == '+' else -1)
            return specie, charge
        else:
            return specie, 0
    else:
        return specie

def atom_to_onehot(atom):
    """Str -> onehot vector"""
    atom = extract_atom_specie(atom)
    vec = torch.zeros(len(ATOM_TYPES))
    if atom in ATOM_TYPES:
        vec[ATOM_TYPES.index(atom)] = 1.0
    else:
        raise KeyError
    return vec


class MaterialsGraphDataset(Dataset):
    """JSON → PyG Data object"""

    def __init__(self, data_dir, target=None, prefixes=None, onehot=False):
        self.data_dir = data_dir
        self.target_key = target
        self.is_onehot = onehot
        self.file_list = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.endswith(".json") and (prefixes is None or any(fname.startswith(p) for p in prefixes))
        ]

    def __len__(self):
        return len(self.file_list)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        path = self.file_list[idx]
        with open(path, "r") as f:
            item = json.load(f)

        material_id = item["material_id"]
        nodes = item["graph"]["nodes"]
        edges = item["graph"]["edges"]
        cart_distances = item["graph"]["cart_dist"]
        cart_directions = item["graph"]["cart_dir"]
        props = item["properties"]

        if self.is_onehot:
            x = torch.stack([atom_to_onehot(atom) for atom in nodes])
        else:
            x = torch.tensor([ATOM_TYPES.index(extract_atom_specie(atom)) for atom in nodes], dtype=torch.long)
            
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(edge_index.size(1), 1)
        cart_dist = torch.tensor(cart_distances, dtype=torch.float)
        cart_dir = torch.tensor(cart_directions, dtype=torch.float)
        non_H_mask = torch.tensor(
            [not extract_atom_specie(atom)=='H' for atom in nodes],
            dtype=torch.bool
        )

        if not self.target_key:
            raise ValueError("self.target_key must contain at least one key.")

        try:
            y_values = [props[k] for k in self.target_key]
        except KeyError as e:
            raise KeyError(f"{material_id} is missing required key: {e}")

        y = torch.tensor([y_values], dtype=torch.float)

        data = Data(
            x=x,
            material_id=material_id,
            edge_index=edge_index,
            edge_attr=edge_attr,
            cart_dist=cart_dist,
            cart_dir=cart_dir,
            atom_types=nodes,
            non_H_mask=non_H_mask,
            y=y
        )
        return data
