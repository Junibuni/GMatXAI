from tqdm.auto import tqdm

import roma
import torch
from jarvis.core.specie import get_node_attributes
from jarvis.core.atoms import Atoms
from torch_geometric.data import Data, Batch
from torch_geometric.data import InMemoryDataset

from src.data.utils import radius_graph_pbc, optmize_lattice
from src.utils.atom_info import MEGNET_ATOM_EMBEDDING

def get_megnet_embedding(symbol: str):
    return torch.tensor(MEGNET_ATOM_EMBEDDING.get(symbol, [0.0] * 16), dtype=torch.float)
class Figshare_Dataset(InMemoryDataset):
    def __init__(self, root, data, targets, transform=None, pre_transform=None, name="jarvis", radius=5.0, max_neigh=-1, augment=False, mode="", mean=1.0, std=0.0, norm=False):
        
        self._input_data = data
        self._input_targets = targets
        self.mode = mode
        self.name = name
        self.radius = radius
        self.max_neigh = max_neigh if max_neigh > 0 else None
        self.augment = augment
        self.norm = norm
        self.mean = mean
        self.std = std
        super(Figshare_Dataset, self).__init__(root, transform, pre_transform)
        self._data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.name + ".pt"

    def download(self):
        pass
    

    def get(self, idx):
        data = super().get(idx)
        
        if self.augment:
            data = self.augment_data(data)
        
        return data

    def augment_data(self, data):
        R = roma.utils.random_rotmat(size=1, device=data.x.device).squeeze(0)    
        data.cart_dir = data.cart_dir @ R
        data.cell = data.cell @ R

        return data

    def process(self):
        data_list = []
        for i, (ddat, target) in tqdm(enumerate(zip(self._input_data, self._input_targets)), total=len(self._input_data), desc=f"{self.mode} data"):
            structure = Atoms.from_dict(ddat["atoms"])
            atomic_numbers = torch.tensor([get_node_attributes(s, atom_features="atomic_number") for s in structure.elements]).squeeze(-1)
            target = torch.tensor(target).view(-1, 1)
            if self.norm:
                target = (target - self.mean) / self.std
            data = Data(x=atomic_numbers, y=target)
            data.pos = torch.tensor(structure.cart_coords, dtype=torch.float32)
            data.cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
            data.pbc = torch.tensor([[True, True, True]])
            data.natoms = torch.tensor([data.x.shape[0]])
            data.cell = data.cell.unsqueeze(0)

            #Compute PBC
            batch = Batch.from_data_list([data])
            edge_index, _, _, cart_vector = radius_graph_pbc(batch, self.radius, self.max_neigh)
            
            data.cart_dist = torch.norm(cart_vector, p=2, dim=-1)
            data.cart_dir = torch.nn.functional.normalize(cart_vector, p=2, dim=-1)
            
            try:
                megnet_embeds = [
                    torch.tensor(MEGNET_ATOM_EMBEDDING[symbol], dtype=torch.float)
                    if symbol in MEGNET_ATOM_EMBEDDING else torch.zeros(16)
                    for symbol in structure.elements
                ]
            except Exception as e:
                raise ValueError(f"MEGNet Embedding Error: {e}")
            
            atom_megnet_embed = torch.stack(megnet_embeds, dim=0)
            data.atom_megnet_embed = atom_megnet_embed

            data.edge_index = edge_index
            data.edge_attr = cart_vector
            delattr(data, "pbc")
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])