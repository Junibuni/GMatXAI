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
        pooling="mean",
        use_edge_features=False
    ):
        super().__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_edge_features = use_edge_features

        self.node_emb = nn.Linear(node_input_dim, hidden_dim)
        self.edge_emb = (
            nn.Linear(edge_input_dim, hidden_dim)
            if use_edge_features and edge_input_dim > 0
            else None
        )

        self.pool = get_pooling_layer(pooling, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.convs = nn.ModuleList()

    def forward(self, data):
        raise NotImplementedError("Subclasses must implement forward()")

    def reset_parameters(self):
        self.node_emb.reset_parameters()
        if self.edge_emb is not None:
            self.edge_emb.reset_parameters()
        for m in self.mlp:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        if hasattr(self.pool, "reset_parameters"):
            self.pool.reset_parameters()

    def get_config(self):
        return {
            "node_input_dim": self.node_input_dim,
            "edge_input_dim": self.edge_input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "pooling": self.pool.__name__ if hasattr(self.pool, "__name__") else str(self.pool),
            "use_edge_features": self.use_edge_features
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_dim={self.hidden_dim}, num_layers={self.num_layers})"

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
            pooling=getattr(config, "pooling", "mean"),
            use_edge_features=getattr(config, "use_edge_features", False)
        )