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

class CartNetBlock(nn.Module):
    def __init__(self, num_layers, hidden_dim, radius, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([CartNet_layer(dim_in=hidden_dim, radius=radius) for _ in range(num_layers)])
        self.norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        for layer, norm in zip(self.layers, self.norms):
            batch = layer(batch)
            h_new = norm(batch.x, batch.batch)
            h = h + self.dropout(h_new)
            batch.x = h
        return batch

class MatformerBlock(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, edge_dim, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([MatformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            edge_dim=edge_dim,
            concat=False,
            beta=True
        ) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        ) for _ in range(num_layers)])
        self.norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch
        for conv, ffn, norm in zip(self.convs, self.ffns, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = ffn(h_new)
            h_new = norm(h_new, batch_idx)
            h = h + self.dropout(h_new)
        batch.x = h
        return batch

class AttentionFusionMixer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score_mlp = nn.Linear(hidden_dim, 1)

    def forward(self, h_A, h_B):
        score_A = self.score_mlp(h_A)
        score_B = self.score_mlp(h_B)
        weights = torch.softmax(torch.cat([score_A, score_B], dim=1), dim=1)
        w_A = weights[:, 0].unsqueeze(1)
        w_B = weights[:, 1].unsqueeze(1)
        h_out = w_A * h_A + w_B * h_B
        return h_out

class UniCrystalFormer(nn.Module):
    def __init__(self,
        num_cart_layers: int = 3,
        num_mat_layers: int = 2,
        edge_features: int = 128,
        hidden_dim: int = 128,
        fc_features: int = 128,
        output_features: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        radius: float = 8.0
    ):
        super().__init__()
        self.num_cart_layers = num_cart_layers
        self.num_mat_layers = num_mat_layers
        
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
        
        # CartNet block
        self.cartnet_block = CartNetBlock(num_cart_layers, hidden_dim, radius, dropout)
        
        # Matformer block
        self.matformer_block = MatformerBlock(num_mat_layers, hidden_dim, num_heads, edge_features, dropout)
        
        # Mixer
        self.mixer = AttentionFusionMixer(hidden_dim)
        
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
        
        # Clone for branches
        batch_cart = data.clone()
        batch_mat = data.clone()
        
        # Apply blocks
        batch_cart = self.cartnet_block(batch_cart)
        batch_mat = self.matformer_block(batch_mat)
        
        # Mix outputs
        h_cart = batch_cart.x
        h_mat = batch_mat.x
        x_out = self.mixer(h_cart, h_mat)
        
        # Readout and final prediction
        features = self.readout(x_out, data.batch)
        out = self.fc_readout(features)
        return out

    @classmethod
    def from_config(cls, config):
        return cls(
            num_cart_layers=config.get("num_cart_layers", 3),
            num_mat_layers=config.get("num_mat_layers", 2),
            edge_features=config.get("edge_features", 128),
            hidden_dim=config.get("hidden_dim", 128),
            fc_features=config.get("fc_features", 128),
            output_features=config.get("output_features", 1),
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.1),
            radius=config.get("radius", 8.0)
        )