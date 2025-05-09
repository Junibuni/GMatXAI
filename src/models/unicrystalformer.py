from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import GraphNorm
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
        
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

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
    def __init__(self, hidden_dim, radius, heads, edge_dim, mix_layers, mixer_type, dropout=0.1, residual_scale=0.5):
        super().__init__()
        self.mix_layers = mix_layers
        self.residual_scale = residual_scale
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
        self.mixer = mixer_type(hidden_dim)

        self.norm_cart = GraphNorm(hidden_dim)
        self.norm_mat = GraphNorm(hidden_dim)

    def forward(self, batch_cart, batch_mat):
        cartnet_batch = self.cartnet(batch_cart)
        x_cart = self.norm_cart(cartnet_batch.x, batch_cart.batch)

        x_mat = self.matformer(batch_mat.x, batch_mat.edge_index, batch_mat.edge_attr)
        x_mat = self.norm_mat(x_mat, batch_mat.batch)

        if self.mix_layers:
            x_out = self.mixer(x_cart, x_mat)
            x_out = self.dropout(x_out)
            batch_cart.x = x_cart + self.residual_scale * x_out
            batch_mat.x = x_mat + self.residual_scale * x_out
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
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        nn.init.kaiming_uniform_(self.atom_embedding.weight, nonlinearity='relu')
        
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
                    edge_dim=edge_features,
                    mix_layers=mix_layers, 
                    mixer_type=self.mixer,
                    dropout=dropout,
                    residual_scale=0.5
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

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
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
