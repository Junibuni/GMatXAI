# Re-import needed modules after kernel reset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool

from src.models.base import BaseGNNModel

class GAT(BaseGNNModel):
    def __init__(self,         
        node_input_dim,
        edge_input_dim,
        hidden_dim,
        num_layers,
        output_dim,
        pooling="mean",
        use_edge_features=True,
        heads=1,
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
        self.heads = heads
        self.concat = True

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(self.num_layers):
            in_dim = self.hidden_dim * self.heads if i > 0 else self.hidden_dim
            out_dim = self.hidden_dim
            self.convs.append(
                GATConv(in_channels=in_dim,
                        out_channels=out_dim,
                        heads=self.heads,
                        concat=self.concat,
                        dropout=0.0)
            )
            self.batch_norms.append(nn.BatchNorm1d(out_dim * self.heads if self.concat else out_dim))

        self.final_dim = self.hidden_dim * self.heads if self.concat else self.hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim),
            nn.Softplus(),
            nn.Linear(self.final_dim, self.output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_emb(x)

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        x = self.pool(x, batch)
        return self.mlp(x)

    def get_config(self):
        config = {
            "node_input_dim": self.node_input_dim,
            "edge_input_dim": self.edge_input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "pooling": "mean",
            "use_edge_features": False,
            "heads": self.heads
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            node_input_dim=config["node_input_dim"],
            edge_input_dim=config["edge_input_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            output_dim=config["output_dim"],
            pooling=config.get("pooling", "mean"),
            use_edge_features=False,
            heads=config.get("heads", 1)
        )
