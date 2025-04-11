import torch
import torch.nn as nn
from torch_geometric.nn import Set2Set
from torch_geometric.utils import scatter

from src.models.base import BaseGNNModel

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation=nn.Softplus):
        super().__init__()
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            input_dim = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MEGNetBlock(nn.Module):
    def __init__(self, hidden_dim, use_u):
        super().__init__()
        self.use_u = use_u
        u_dim = hidden_dim if use_u else 0

        self.phi_e = MLP(2 * hidden_dim + hidden_dim + u_dim, [hidden_dim, hidden_dim])
        self.phi_v = MLP(hidden_dim + hidden_dim + u_dim, [hidden_dim, hidden_dim])
        self.phi_u = MLP(hidden_dim + hidden_dim + hidden_dim, [hidden_dim, hidden_dim]) if use_u else None

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        e_input = [x[row], x[col], edge_attr]
        if self.use_u:
            e_input.append(u[batch[row]])
        edge_attr = self.phi_e(torch.cat(e_input, dim=-1))

        e_aggr = scatter(edge_attr, row, dim=0, reduce='mean')
        v_input = [x, e_aggr]
        if self.use_u:
            v_input.append(u[batch])
        x = self.phi_v(torch.cat(v_input, dim=-1))

        if self.use_u:
            e_mean = scatter(edge_attr, batch[row], dim=0, reduce='mean')
            v_mean = scatter(x, batch, dim=0, reduce='mean')
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
        pooling="mean",  # unused, Set2Set is fixed
        use_edge_features=True,
        global_input_dim=0,
    ):
        super().__init__(
            node_input_dim,
            edge_input_dim,
            hidden_dim,
            num_layers,
            output_dim,
            pooling=pooling,
            use_edge_features=use_edge_features,
        )

        self.use_u = global_input_dim > 0
        self.global_input_dim = global_input_dim
        self.global_emb = nn.Linear(global_input_dim, hidden_dim) if self.use_u else None

        self.node_emb = nn.Linear(node_input_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_input_dim, hidden_dim) if use_edge_features else None

        self.convs = nn.ModuleList([
            MEGNetBlock(hidden_dim, use_u=self.use_u) for _ in range(num_layers)
        ])

        self.set2set_v = Set2Set(hidden_dim, processing_steps=3)
        self.set2set_e = Set2Set(hidden_dim, processing_steps=3)

        final_dim = 2 * hidden_dim + 2 * hidden_dim
        if self.use_u:
            final_dim += hidden_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.Softplus(),
            nn.Linear(32, 16),
            nn.Softplus(),
            nn.Linear(16, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if self.use_edge_features else torch.zeros((edge_index.size(1), 1), device=x.device)
        u = getattr(data, 'u', None)
        if self.use_u:
            assert u is not None, "[MEGNet] global_input_dim > 0, but data.u is None"
            u = self.global_emb(u)

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr) if self.use_edge_features else edge_attr

        for block in self.convs:
            x, edge_attr, u = block(x, edge_index, edge_attr, u, batch)

        v_pool = self.set2set_v(x, batch)
        e_pool = self.set2set_e(edge_attr, batch[edge_index[0]])

        out = [v_pool, e_pool]
        if self.use_u:
            out.append(u)
        return self.final_mlp(torch.cat(out, dim=-1))

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
            pooling=config.get("pooling", "mean"),
            use_edge_features=config.get("use_edge_features", False),
            global_input_dim=config.get("global_input_dim", 0)
        )