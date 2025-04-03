import torch
import torch.nn as nn

from src.models.pooling import get_pooling_layer

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

        self.pool = get_pooling_layer(pooling, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.convs = nn.ModuleList()

    def forward(self, data):
        raise NotImplementedError("Subclasses must implement forward()")

    @classmethod
    def from_config(cls, config):
        required_fields = [
            "node_input_dim", "edge_input_dim", "hidden_dim", "num_layers", "output_dim"
        ]
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"[Config Error] '{field}' is missing in model config.")

        return cls(
            node_input_dim=config.node_input_dim,
            edge_input_dim=config.edge_input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_dim=config.output_dim,
            pooling=getattr(config, "pooling", "mean")
        )