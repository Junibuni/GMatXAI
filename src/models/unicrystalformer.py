from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import GraphNorm, Set2Set
from torch_scatter import scatter

from src.models.cartnet import CartNet_layer
from src.models.matformer.utils import RBFExpansion
from src.models.matformer.transformer import MatformerConv

class ResidualGateMixer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualGateMixer, self).__init__()
        self.mlp_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_A, h_B, **kwargs):
        assert h_A.shape == h_B.shape, "h_A and h_B must have the same shape"

        gate = self.mlp_gate(torch.cat([h_A, h_B], dim=-1))
        mixed = gate * h_A + (1 - gate) * h_B
        return self.norm(self.dropout(mixed))

class CrossMixAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CrossMixAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, h_A, h_B, **kwargs):
        h_A = h_A.unsqueeze(0)
        h_B = h_B.unsqueeze(0)
        h_mixed, _ = self.attention(h_A, h_B, h_B)
        return h_mixed.squeeze(0)

# class CrossMixAttention(nn.Module):
#     def __init__(self, hidden_dim, dropout=0.1):
#         super(CrossMixAttention, self).__init__()
#         self.query = nn.Linear(hidden_dim, hidden_dim)
#         self.key = nn.Linear(hidden_dim, hidden_dim)
#         self.value = nn.Linear(hidden_dim, hidden_dim)
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(dropout)
        
#         nn.init.xavier_uniform_(self.query.weight)
#         nn.init.xavier_uniform_(self.key.weight)
#         nn.init.xavier_uniform_(self.value.weight)

#     def forward(self, h_A, h_B):
#         Q = self.query(h_A)   # [n_nodes, hidden_dim]
#         K = self.key(h_B)     # [n_nodes, hidden_dim]
#         V = self.value(h_B)   # [n_nodes, hidden_dim]

#         attn_scores = (Q * K).sum(dim=-1, keepdim=True) / (Q.size(-1) ** 0.5)
#         attn_weights = torch.sigmoid(attn_scores)

#         attended = attn_weights * V  # [n_nodes, hidden_dim]
#         fused = self.norm(attended + h_A)  # Residual + LayerNorm
#         fused = self.dropout(fused)

#         return fused

class MoESoftRoutingMixer(nn.Module):
    def __init__(self, d_model):
        super(MoESoftRoutingMixer, self).__init__()
        self.mlp_router = nn.Sequential(
            nn.Linear(2 * d_model, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, h_A, h_B, **kwargs):
        router_input = torch.cat([h_A, h_B], dim=-1)
        weights = self.mlp_router(router_input)  # [num_nodes, 2]
        w_A = weights[:, 0].unsqueeze(-1)
        w_B = weights[:, 1].unsqueeze(-1)
        return w_A * h_A + w_B * h_B

class UniCrystalFormerLayer(nn.Module):
    def __init__(self, hidden_dim, radius, heads, edge_dim, mix_layers, mixer, dropout=0.1, residual_scale=0.5):
        super().__init__()
        self.mix_layers = mix_layers
        self.residual_scale_cart = nn.Parameter(torch.tensor(0.5))
        self.residual_scale_mat = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        
        self.cartnet = CartNet_layer(dim_in=hidden_dim, radius=radius)
        self.matformer = MatformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=False,
            beta=True
        )

        self.matformer_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.mixer = mixer

        self.norm_cart = GraphNorm(hidden_dim)
        self.norm_mat = GraphNorm(hidden_dim)
        self.norm_pre_mix_cart = nn.LayerNorm(hidden_dim)
        self.norm_pre_mix_mat = nn.LayerNorm(hidden_dim)

    def forward(self, batch_cart, batch_mat):
        cartnet_batch = self.cartnet(batch_cart)
        x_cart = self.norm_cart(cartnet_batch.x, batch_cart.batch)

        x_mat = self.matformer(batch_mat.x, batch_mat.edge_index, batch_mat.edge_attr)
        x_mat = self.matformer_ffn(x_mat)
        x_mat = self.norm_mat(x_mat, batch_mat.batch)

        if self.mix_layers:
            x_cart_norm = self.norm_pre_mix_cart(x_cart)
            x_mat_norm = self.norm_pre_mix_mat(x_mat)
            x_out = self.mixer(x_cart_norm, x_mat_norm, edge_attr=batch_mat.edge_attr, batch=batch_mat.batch)
            x_out = self.dropout(x_out)
            batch_cart.x = x_cart + self.residual_scale_cart * x_out
            batch_mat.x = x_mat + self.residual_scale_mat * x_out
        else:
            batch_cart.x = x_cart
            batch_mat.x = x_mat
        
        return batch_cart, batch_mat

class AtomEncoder(nn.Module):
    def __init__(self, num_atom_types, hidden_dim, dropout=0.1):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)         # atomic number 기반
        self.megnet_proj = nn.Linear(16, hidden_dim)             # MEGNet 임베딩

        nn.init.kaiming_uniform_(self.atom_embedding.weight, nonlinearity='relu')

        self.gate_layer = nn.Linear(2 * hidden_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, atomic_numbers, megnet_embed):
        emb = self.atom_embedding(atomic_numbers)            # [N, hidden_dim]
        megnet = self.megnet_proj(megnet_embed)         # [N, hidden_dim]

        combined = torch.cat([emb, megnet], dim=-1)
        gate = torch.sigmoid(self.gate_layer(combined))
        fused = gate * emb + (1 - gate) * megnet
        return self.out_proj(fused)
    
class UniCrystalFormer(nn.Module):
    def __init__(self,
        conv_layers: int = 5,
        edge_features: int = 128,
        hidden_dim: int = 128,
        fc_features: int = 128,
        output_features: int = 1,
        num_heads: int = 4,
        mix_layers: bool = True,
        mixer_type: Literal["residual_gate", "cross_attention", "moe_soft_routing"] = 'residual_gate',
        dropout: float = 0.1,
        radius: float = 8.0
        ):
        super().__init__()
        self.conv_layers = conv_layers
        self.mix_layers = mix_layers
        # Atom embedding
        num_atom_types = 119
        self.atom_embedding = AtomEncoder(num_atom_types, hidden_dim, dropout)
        
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=edge_features,
            ),
            nn.Linear(edge_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        if mixer_type == 'residual_gate':
            mixer_ctor = lambda: ResidualGateMixer(hidden_dim, dropout)
        # elif mixer_type == 'cross_attention':
        #     self.mixer = CrossAttentionMixer(hidden_dim, num_heads)
        elif mixer_type == 'cross_attention':
            mixer_ctor = lambda: CrossMixAttention(hidden_dim, num_heads)
        elif mixer_type == 'moe_soft_routing':
            mixer_ctor = lambda: MoESoftRoutingMixer(hidden_dim)
        else:
            raise ValueError(f"Invalid mixer_type {mixer_type}")
        
        self.att_layers = nn.ModuleList(
            [
                UniCrystalFormerLayer(
                    hidden_dim, 
                    radius, 
                    num_heads, 
                    edge_dim=edge_features,
                    mix_layers=mix_layers, 
                    mixer=mixer_ctor(),
                    dropout=dropout,
                    residual_scale=0.5
                )
                for _ in range(conv_layers)
            ]
        )
        self.mixer = mixer_ctor()
        self.readout = Set2Set(hidden_dim, processing_steps=3)  # usually 3-6 is enough

        self.fc_readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, fc_features),  # doubled due to Set2Set
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
        node_features = self.atom_embedding(data.x, data.atom_megnet_embed)
        edge_dist = torch.norm(data.edge_attr, dim=1)
        edge_attr = self.rbf(edge_dist)
        
        data.x = node_features
        data.edge_attr = edge_attr
        
        batch_cart = data.clone()
        batch_mat = data.clone()
        for i in range(self.conv_layers):
            batch_cart, batch_mat = self.att_layers[i](batch_cart, batch_mat)

        if not self.mix_layers:
            x_cart = batch_cart.x
            x_mat = batch_mat.x
            x_out = self.mixer(x_cart, x_mat, edge_attr=batch_mat.edge_attr, batch=batch_mat.batch)
        else:
            x_out = (batch_cart.x + batch_mat.x) / 2

        # crystal-level readout
        features = self.readout(x_out, data.batch)
        out = self.fc_readout(features)

        return out

    @classmethod
    def from_config(cls, config):
        return cls(
            conv_layers = config.get("conv_layers", 5),
            edge_features = config.get("edge_features", 128),
            hidden_dim = config.get("hidden_dim", 128),
            fc_features = config.get("fc_features", 128),
            output_features = config.get("output_features", 1),
            num_heads = config.get("num_heads", 4),
            mix_layers = config.get("mix_layers", True),
            mixer_type = config.get("mixer_type", 'residual_gate'),
            dropout = config.get("dropout", 0.1),
            radius = config.get("radius", 8.0)
        )
