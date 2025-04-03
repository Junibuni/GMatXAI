import torch.nn as nn
from torch_geometric.data import Data

class CGCNNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        return self.model(data)
