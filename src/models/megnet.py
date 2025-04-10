import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from src.models.base import BaseGNNModel

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation=nn.ReLU):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(activation())
            input_dim = dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class MEGNetBlock(nn.Module):
    def __init__(self, dim_e, dim_v, dim_u):
        super().__init__()
        self.phi_e = MLP(dim_e + 2 * dim_v + dim_u, [dim_e, dim_e])
        self.phi_v = MLP(dim_v + dim_e + dim_u, [dim_v, dim_v])
        self.phi_u = MLP(dim_u + dim_e + dim_v, [dim_u, dim_u])

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        u_e = u[batch[row]]
        e_input = torch.cat([x[row], x[col], edge_attr, u_e], dim=-1)
        edge_attr = self.phi_e(e_input)

        e_aggr = scatter(edge_attr, row, dim=0, reduce="mean")
        u_v = u[batch]
        v_input = torch.cat([x, e_aggr, u_v], dim=-1)
        x = self.phi_v(v_input)

        e_mean = scatter(edge_attr, batch[row], dim=0, reduce="mean")
        v_mean = scatter(x, batch, dim=0, reduce="mean")
        u_input = torch.cat([u, e_mean, v_mean], dim=-1)
        u = self.phi_u(u_input)

        return x, edge_attr, u

class MEGNet(BaseGNNModel):
    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        hidden_dim,
        num_layers,
        output_dim,
        pooling="mean",  # unused, set2set is fixed
        use_edge_features=True,
        global_input_dim=2  # configurable global u
    ):
        super().__init__(
            node_input_dim, 
            edge_input_dim, 
            hidden_dim, 
            num_layers, 
            output_dim, 
            pooling, 
            use_edge_features
        )

        self.global_input_dim = global_input_dim
        self.global_emb = nn.Linear(global_input_dim, hidden_dim)

        self.convs = nn.ModuleList([
            MEGNetBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.set2set_e = Set2Set(hidden_dim, processing_steps=3)
        self.set2set_v = Set2Set(hidden_dim, processing_steps=3)

        self.final_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2 * hidden_dim + hidden_dim, 32),
            nn.Softplus(),
            nn.Linear(32, 16),
            nn.Softplus(),
            nn.Linear(16, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if self.use_edge_features else torch.zeros((edge_index.size(1), 1), device=x.device)
        u = data.u  # [batch_size, global_input_dim]

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr) if self.edge_emb else edge_attr
        u = self.global_emb(u)

        for conv in self.convs:
            x, edge_attr, u = conv(x, edge_index, edge_attr, u, batch)

        e_pool = self.set2set_e(edge_attr, batch[edge_index[0]])
        v_pool = self.set2set_v(x, batch)

        out = torch.cat([e_pool, v_pool, u], dim=1)
        return self.final_mlp(out)

    def get_config(self):
        config = super().get_config()
        config["global_input_dim"] = self.global_input_dim
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            node_input_dim=config.node_input_dim,
            edge_input_dim=config.edge_input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_dim=config.output_dim,
            pooling=getattr(config, "pooling", "mean"),
            use_edge_features=getattr(config, "use_edge_features", False),
            global_input_dim=getattr(config, "global_input_dim", 2)
        )