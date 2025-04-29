import numpy as np
import torch
import torch.nn.functional as F
from typing import Literal
from torch import nn
from torch_scatter import scatter

from src.models.cartnet import CartNet_layer
from src.models.matformer.utils import RBFExpansion
from src.models.matformer.transformer import MatformerConv

class CrossMix(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x_c, x_m):
        assert x_c.shape == x_m.shape, "x_c and x_m must have the same shape"
        
        x = torch.cat([x_c, x_m], dim=-1)
        gate = torch.sigmoid(self.gate(x))
        fused = self.linear(x)
        return gate * fused + (1 - gate) * (x_c + x_m) / 2
    
class UniCrystalFormerLayer(nn.Module):
    def __init__(self, hidden_dim, radius, heads, edge_dim):
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
        self.crossmix = CrossMix(hidden_dim)

    def forward(self, batch):
        x_c = batch.x

        batch = self.cartnet(batch)
        x_cart = batch.x

        x_mat = self.matformer(x_c, batch.edge_index, batch.edge_attr)

        x_out = self.crossmix(x_cart, x_mat)

        batch.x = x_out + x_c
        return batch

class UniCrystalFormer(nn.Module):
    def __init__(self,
        conv_layers: int = 5,
        atom_input_features: int = 92,
        edge_features: int = 128,
        node_features: int = 128,
        fc_features: int = 128,
        output_features: int = 1,
        node_layer_head: int = 4,
        link: Literal["identity", "log", "logit"] = "identity",
        zero_inflated: bool = False,
        ):
        super().__init__()
        self.zero_inflated = zero_inflated
        self.conv_layers = conv_layers
        # CGCNN style atom embedding
        self.atom_embedding = nn.Linear(
            atom_input_features, node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=edge_features,
            ),
            nn.Linear(edge_features, node_features),
            nn.Softplus(),
            nn.Linear(node_features, node_features),
        )

        self.att_layers = nn.ModuleList(
            [
                MatformerConv(in_channels=node_features, out_channels=node_features, heads=node_layer_head, edge_dim=node_features)
                for _ in range(conv_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(node_features, fc_features), 
            nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()

        self.fc_out = nn.Linear(
            fc_features, output_features
        )

        self.link = None
        self.link_name = link
        if link == "identity":
            self.link = lambda x: x
        elif link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            if not self.zero_inflated:
                self.fc_out.bias.data = torch.tensor(
                    np.log(avg_gap), dtype=torch.float
                )

    def forward(self, data) -> torch.Tensor:
            
        node_features = self.atom_embedding(data.x)
        edge_feat = torch.norm(data.edge_attr, dim=1)
        
        edge_features = self.rbf(edge_feat)
        
        for i in range(self.conv_layers):
            node_features = self.att_layers[i](node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")
        
        features = self.fc(features)

        out = self.fc_out(features)
        if self.link:
            out = self.link(out)

        return torch.squeeze(out)


