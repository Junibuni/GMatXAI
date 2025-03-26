import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops
class CGCNNConv(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super(CGCNNConv, self).__init__(aggr="add")
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # edge_index: [2, E], edge_attr: [E, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: center node, x_j: neighbor, edge_attr: [E, edge_dim]
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.edge_mlp(msg_input)


class CGCNN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_layers, output_dim):
        super(CGCNN, self).__init__()
        self.node_emb = nn.Linear(node_input_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_input_dim, hidden_dim)

        self.convs = nn.ModuleList([
            CGCNNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        self.pool = global_mean_pool

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)

        x = self.pool(x, batch)
        out = self.mlp(x)
        return out
