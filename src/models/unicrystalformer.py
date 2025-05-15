from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import GraphNorm, Set2Set

from src.models.cartnet import CartNet_layer
from src.models.matformer.utils import RBFExpansion
from src.models.matformer.transformer import MatformerConv

class AtomEncoder(nn.Module):
    def __init__(self, num_atom_types, hidden_dim, dropout=0.1):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.megnet_proj = nn.Linear(16, hidden_dim)
        self.gate_layer = nn.Linear(2 * hidden_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, atomic_numbers, megnet_embed):
        emb = self.atom_embedding(atomic_numbers)
        megnet = self.megnet_proj(megnet_embed)
        combined = torch.cat([emb, megnet], dim=-1)
        gate = torch.sigmoid(self.gate_layer(combined))
        fused = gate * emb + (1 - gate) * megnet
        return self.out_proj(fused)

class UniLayer(nn.Module):
    def __init__(self, hidden_dim, radius, heads, edge_dim, dropout=0.1):
        super().__init__()
        self.cartnet = CartNet_layer(dim_in=hidden_dim, radius=radius)
        self.matformer = MatformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=False,
            beta=True
        )
        self.norm_cart = GraphNorm(hidden_dim)
        self.norm_mat = GraphNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch

        batch_cart = self.cartnet(batch)
        h_cart = self.norm_cart(batch_cart.x, batch_idx)
        h = h + self.dropout(h_cart)

        h_mat = self.matformer(h, edge_index, edge_attr)
        h_mat = self.norm_mat(h_mat, batch_idx)
        h = h + self.dropout(h_mat)

        batch.x = h
        return batch

class UniCrystalFormer(nn.Module):
    def __init__(self,
        conv_layers: int = 3,
        edge_features: int = 128,
        hidden_dim: int = 128,
        fc_features: int = 128,
        output_features: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        radius: float = 8.0
    ):
        super().__init__()
        self.conv_layers = conv_layers
        
        # Atom embedding
        num_atom_types = 119
        self.atom_embedding = AtomEncoder(num_atom_types, hidden_dim, dropout)
        
        # Edge feature expansion
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=edge_features),
            nn.Linear(edge_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Convolutional layers
        self.uni_layers = nn.ModuleList(
            [
                UniLayer(
                    hidden_dim,
                    radius,
                    num_heads,
                    edge_dim=edge_features,
                    dropout=dropout
                )
                for _ in range(conv_layers)
            ]
        )
        
        # Readout and final FC layers
        self.readout = Set2Set(hidden_dim, processing_steps=3)
        self.fc_readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, fc_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_features, output_features)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, data) -> torch.Tensor:
        # Encode atom and edge features
        node_features = self.atom_embedding(data.x, data.atom_megnet_embed)
        edge_dist = torch.norm(data.edge_attr, dim=1)
        edge_attr = self.rbf(edge_dist)
        
        # Update data with encoded features
        data.x = node_features
        data.edge_attr = edge_attr
        
        # Apply convolutional layers
        for layer in self.uni_layers:
            data = layer(data)
        
        # Readout and final prediction
        x_out = data.x
        features = self.readout(x_out, data.batch)
        out = self.fc_readout(features)
        return out

    @classmethod
    def from_config(cls, config):
        return cls(
            conv_layers=config.get("conv_layers", 3),
            edge_features=config.get("edge_features", 128),
            hidden_dim=config.get("hidden_dim", 128),
            fc_features=config.get("fc_features", 128),
            output_features=config.get("output_features", 1),
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.1),
            radius=config.get("radius", 8.0)
        )