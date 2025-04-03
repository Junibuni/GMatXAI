import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool

class GlobalAttentionPool(nn.Module):
    def __init__(self, gate_nn):
        super().__init__()
        self.gate_nn = gate_nn

    def forward(self, x, batch):
        gate = torch.sigmoid(self.gate_nn(x))
        return torch.zeros_like(x).scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x * gate)


def get_pooling_layer(method="mean", hidden_dim=64):
    if method == "mean":
        return global_mean_pool
    elif method == "sum":
        return global_add_pool
    elif method == "attn":
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        return GlobalAttentionPool(gate_nn)
    else:
        raise ValueError(f"Unknown pooling method: {method}")
