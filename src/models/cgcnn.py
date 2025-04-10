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

    def forward(self, x, edge_index, edge_attr=None):
        # x: [num_nodes, node_fea_len]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_fea_len]
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: target node features [num_edges, node_fea_len]
        # x_j: source node features [num_edges, node_fea_len]
        # edge_attr: edge features [num_edges, edge_fea_len]
        if edge_attr is None:
            edge_attr = torch.zeros(x_i.size(0), self.edge_fea_len, device=x_i.device)
        
        z = torch.cat([x_i, x_j, edge_attr], dim=1)  # [num_edges, 2*node_fea_len + edge_fea_len]
        gate = self.message_mlp(z)
        msg = self.gate_mlp(z)
        return gate * msg  # element-wise product

    def update(self, aggr_out, x):
        return F.softplus(x + aggr_out)


class CGCNN(BaseGNNModel):
    def __init__(
        self, 
        node_input_dim,
        edge_input_dim,
        hidden_dim,
        num_layers,
        output_dim,
        pooling="mean",
        use_edge_features=False
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

        self.convs = nn.ModuleList([
            CGCNNConv(self.hidden_dim, self.edge_input_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_dim)
            for _ in range(self.num_layers)
        ])

    def forward(self, data):
        x = self.node_emb(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr if self.use_edge_features else None
        batch = data.batch

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)

        x = self.pool(x, batch)
        out = self.mlp(x)
        return out