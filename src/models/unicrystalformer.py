from typing import Literal

import torch
from torch import nn
from torch_geometric.data import Data
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
                 num_mat_layers: int = 3,
                 edge_features: int = 128,    # RBF expansion size (bins)
                 hidden_dim: int = 128,
                 fc_features: int = 128,
                 output_features: int = 1,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 radius: float = 8.0):
        super().__init__()
        # 1. Atom feature encoding (learned embedding + MEGNet features with gating)
        num_atom_types = 119  # number of atomic elements possible
        self.atom_encoder = AtomEncoder(num_atom_types, hidden_dim, dropout)
        # 2. Edge (distance) encoding: RBF expansion -> hidden_dim
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0.0, vmax=radius, bins=edge_features),
            nn.Linear(edge_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 3. Define CartNet and TransformerConv layers
        self.cart_layers = nn.ModuleList([
            CartNet_layer(dim_in=hidden_dim, radius=radius) 
            for _ in range(num_cart_layers)
        ])
        self.mat_layers = nn.ModuleList([
            MatformerConv(in_channels=hidden_dim, out_channels=hidden_dim,
                          heads=num_heads, edge_dim=edge_features,
                          concat=False, beta=True)
            for _ in range(num_mat_layers)
        ])
        # 4. Define fusion+FFN sublayers for each hybrid layer
        self.num_layers = max(num_cart_layers, num_mat_layers)
        self.norm_attn = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(self.num_layers)])
        self.ffn_layers = nn.ModuleList([nn.Sequential(
                                            nn.Linear(hidden_dim, 2*hidden_dim),
                                            nn.SiLU(),
                                            nn.Linear(2*hidden_dim, hidden_dim)
                                         ) for _ in range(self.num_layers)])
        self.norm_ffn = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(dropout)
        # 5. Readout: Set2Set for graph-level pooling, then fully-connected output
        self.readout = Set2Set(hidden_dim, processing_steps=3)
        self.fc_out = nn.Sequential(
            nn.Linear(2 * hidden_dim, fc_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_features, output_features)
        )
        # Initialize weights (especially for linear layers) 
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, data):
        # Encode atomic features (atomic number and MEGNet embedding)
        h0 = self.atom_encoder(data.x, data.atom_megnet_embed)         # shape [N, hidden_dim]
        # Encode edge attributes (here data.edge_attr is assumed to contain interatomic displacement or distance vector)
        # We use the norm of the displacement as the distance.
        edge_vec = data.edge_attr                                       # shape [E, 3] if containing displacement vectors
        edge_dist = torch.norm(edge_vec, dim=1)                         # shape [E]
        edge_attr = self.rbf(edge_dist)                                 # shape [E, hidden_dim]
        # Prepare graph index for normalization
        batch_idx = data.batch                                          # shape [N]
        # Initialize features and edges for iterative updates
        h = h0
        edge_index = data.edge_index
        cart_dist = data.cart_dist
        # We will update data.x in-place for local conv usage, but keep a separate 'h' for global usage.
        data.x = h
        data.edge_attr = edge_attr
        # Hybrid layer-by-layer propagation
        for i in range(self.num_layers):
            # Local CartNet update (if layer exists)
            if i < len(self.cart_layers):
                # Clone data to avoid altering original before global attention
                local_data = Data(x=h, edge_index=edge_index, edge_attr=edge_attr, batch=batch_idx, cart_dist=cart_dist)
                local_data = self.cart_layers[i](local_data)   # performs message passing using radius neighbors
                h_local = local_data.x                         # shape [N, hidden_dim]
            else:
                h_local = None
            # Global TransformerConv update (if layer exists)
            if i < len(self.mat_layers):
                h_global = self.mat_layers[i](h, edge_index, edge_attr)  # shape [N, hidden_dim]
            else:
                h_global = None
            # Fuse local and global outputs
            if h_local is not None and h_global is not None:
                h_comb = h_local + h_global
            elif h_local is not None:
                h_comb = h_local
            else:
                h_comb = h_global
            # Apply normalization and first residual connection (attention sub-layer)
            h_comb_norm = self.norm_attn[i](h_comb, batch_idx)          # GraphNorm over combined message
            h_attn = h + self.dropout(h_comb_norm)                      # add skip connection from input h
            # Feed-forward network on fused features
            h_ffn = self.ffn_layers[i](h_attn)                          # transform features
            h_ffn_norm = self.norm_ffn[i](h_ffn, batch_idx)             # normalize FFN output
            h = h_attn + self.dropout(h_ffn_norm)                       # second skip connection
            # Prepare for next layer (h updated). Also update data.x for next CartNet layer usage.
            data.x = h
        # Graph-level readout: aggregate node embeddings to predict formation energy per graph
        graph_emb = self.readout(h, batch_idx)          # Set2Set outputs [batch_graphs, 2*hidden_dim]
        out = self.fc_out(graph_emb)                    # final MLP to scalar prediction
        return out

    @classmethod
    def from_config(cls, config: dict):
        """Factory method to create model from a configuration dictionary."""
        return cls(
            num_cart_layers = config.get("num_cart_layers", 3),
            num_mat_layers  = config.get("num_mat_layers", 3),
            edge_features   = config.get("edge_features", 128),
            hidden_dim      = config.get("hidden_dim", 128),
            fc_features     = config.get("fc_features", 128),
            output_features = config.get("output_features", 1),
            num_heads       = config.get("num_heads", 4),
            dropout         = config.get("dropout", 0.1),
            radius          = config.get("radius", 8.0)
        )
