import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class BaseGNNModel(nn.Module):
    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        hidden_dim,
        num_layers,
        output_dim,
        pooling="mean"
    ):
        super().__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.node_emb = nn.Linear(node_input_dim, hidden_dim)

        self.pool = self._get_pooling(pooling)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.convs = nn.ModuleList()

    def _get_pooling(self, method):
        if method == "mean":
            return global_mean_pool
        # TODO: max/sum 등 추가 가능
        raise ValueError(f"Unsupported pooling method: {method}")

    def forward(self, data):
        raise NotImplementedError("Subclasses must implement forward()")
