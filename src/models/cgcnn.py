import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, scatter

from src.models.base import BaseGNNModel

class CGCNNConv(MessagePassing):
    def __init__(self, node_fea_len, edge_fea_len, out_fea_len):
        super(CGCNNConv, self).__init__(aggr='add')
        self.edge_fea_len = edge_fea_len

        # φ
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_fea_len + edge_fea_len, out_fea_len),
            nn.Sigmoid()
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * node_fea_len + edge_fea_len, out_fea_len),
            nn.Softplus()
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, node_fea_len]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_fea_len]
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: target node features [num_edges, node_fea_len]
        # x_j: source node features [num_edges, node_fea_len]
        # edge_attr: edge features [num_edges, edge_fea_len]
        z = torch.cat([x_i, x_j, edge_attr], dim=1)  # [num_edges, 2*node_fea_len + edge_fea_len]
        gate = self.message_mlp(z)
        msg = self.gate_mlp(z)
        return gate * msg  # element-wise product

    def update(self, aggr_out, x):
        return F.softplus(x + aggr_out)


class CGCNN(BaseGNNModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.convs = nn.ModuleList([
            CGCNNConv(self.hidden_dim, self.edge_input_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_dim)
            for _ in range(self.num_layers)
        ])

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.node_emb(x)

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = torch.relu(x)

        x = self.pool(x, batch)
        return self.mlp(x)