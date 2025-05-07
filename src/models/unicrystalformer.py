import numpy as np
import torch
import torch.nn.functional as F
from typing import Literal
from torch import nn
from torch_scatter import scatter

from src.models.cartnet import CartNet_layer
from src.models.matformer.utils import RBFExpansion
from src.models.matformer.transformer import MatformerConv

class ResidualGateMixer(nn.Module):
    def __init__(self, d_model):
        super(ResidualGateMixer, self).__init__()
        self.mlp_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, h_A, h_B):
        assert h_A.shape == h_B.shape, "h_A and h_B must have the same shape"

        gate = self.mlp_gate(torch.cat([h_A, h_B], dim=-1))
        return gate * h_A + (1 - gate) * h_B

# class CrossAttentionMixer(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super(CrossAttentionMixer, self).__init__()
#         self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

#     def forward(self, h_A, h_B):
#         h_A = h_A.unsqueeze(0)
#         h_B = h_B.unsqueeze(0)
#         h_mixed, _ = self.attention(h_A, h_B, h_B)
#         return h_mixed.squeeze(0)

class CrossMixAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(CrossMixAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_A, h_B):
        Q = self.query(h_A)   # [n_nodes, hidden_dim]
        K = self.key(h_B)     # [n_nodes, hidden_dim]
        V = self.value(h_B)   # [n_nodes, hidden_dim]

        attn_scores = (Q * K).sum(dim=-1, keepdim=True) / (Q.size(-1) ** 0.5)
        attn_weights = torch.sigmoid(attn_scores)

        attended = attn_weights * V  # [n_nodes, hidden_dim]
        fused = self.norm(attended + h_A)  # Residual + LayerNorm
        fused = self.dropout(fused)

        return fused


class MoESoftRoutingMixer(nn.Module):
    def __init__(self, d_model):
        super(MoESoftRoutingMixer, self).__init__()
        self.mlp_router = nn.Sequential(
            nn.Linear(2 * d_model, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, h_A, h_B):
        router_input = torch.cat([h_A, h_B], dim=-1)
        weights = self.mlp_router(router_input)  # [num_nodes, 2]
        w_A = weights[:, 0].unsqueeze(-1)
        w_B = weights[:, 1].unsqueeze(-1)
        return w_A * h_A + w_B * h_B
class UniCrystalFormerLayer(nn.Module):
    def __init__(self, hidden_dim, radius, heads, edge_dim, mix_layers, mixer_type):
        super().__init__()
        self.mix_layers = mix_layers
        
        self.cartnet = CartNet_layer(dim_in=hidden_dim, radius=radius)
        self.matformer = MatformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=False,
            beta=True
        )
        self.mixer = mixer_type(hidden_dim)

    def forward(self, batch_cart, batch_mat):
        cartnet_batch = self.cartnet(batch_cart)
        x_cart = cartnet_batch.x

        x_mat = self.matformer(batch_mat.x, batch_mat.edge_index, batch_mat.edge_attr)

        if self.mix_layers:
            x_out = self.mixer(x_cart, x_mat)
            batch_cart.x = x_out + x_cart
            batch_mat.x = x_out + x_mat
        else:
            batch_cart.x = x_cart
            batch_mat.x = x_mat
        
        return batch_cart, batch_mat

class UniCrystalFormer(nn.Module):
    def __init__(self,
        conv_layers: int = 5,
        edge_features: int = 128,
        hidden_dim: int = 128,
        fc_features: int = 128,
        output_features: int = 1,
        num_heads: int = 4,
        zero_inflated: bool = False,
        mix_layers: bool = True,
        mixer_type: Literal["residual_gate", "cross_attention", "moe_soft_routing"] = 'residual_gate',
        dropout=0.1,
        radius=8.0
        ):
        super().__init__()
        self.zero_inflated = zero_inflated
        self.conv_layers = conv_layers
        # Atom embedding
        num_atom_types = 119
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        torch.nn.init.xavier_uniform_(self.atom_embedding.weight.data)
        
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=edge_features,
            ),
            nn.Linear(edge_features, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        if mixer_type == 'residual_gate':
            self.mixer = ResidualGateMixer
        # elif mixer_type == 'cross_attention':
        #     self.mixer = CrossAttentionMixer(hidden_dim, num_heads)
        elif mixer_type == 'cross_attention':
            self.mixer = CrossMixAttention
        elif mixer_type == 'moe_soft_routing':
            self.mixer = MoESoftRoutingMixer
        else:
            raise ValueError(f"Invalid mixer_type {mixer_type}")
        
        self.att_layers = nn.ModuleList(
            [
                UniCrystalFormerLayer(
                    hidden_dim, 
                    radius, 
                    num_heads, 
                    edge_features=edge_features,
                    mix_layers=mix_layers, 
                    mixer_type=self.mixer
                )
                for _ in range(conv_layers)
            ]
        )

        self.fc_readout = nn.Sequential(
            nn.Linear(hidden_dim, fc_features), 
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_features, output_features)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, data) -> torch.Tensor:
        node_features = self.atom_embedding(data.x)
        edge_feat = torch.norm(data.edge_attr, dim=1)
        edge_features = self.rbf(edge_feat)
        
        data.x = node_features
        data.edge_attr = edge_features
        
        batch_cart = data.clone()
        batch_mat = data.clone()
        for i in range(self.conv_layers):
            batch_cart, batch_mat = self.att_layers[i](batch_cart, batch_mat)

        if not self.mix_layers:
            x_cart = batch_cart.x
            x_mat = batch_mat.x
            x_out = self.mixer(x_cart, x_mat)
        else:
            x_out = (batch_cart.x + batch_mat.x) / 2

        # crystal-level readout
        features = scatter(x_out, data.batch, dim=0, reduce="mean")
        
        out = self.fc_readout(features)

        return torch.squeeze(out)


