import torch

class ModelWrapper(torch.nn.Module):
    def __init__(self, base_model, original_data):
        super().__init__()
        self.base_model = base_model
        self.original_data = original_data

    def forward(self, x, edge_index):
        d = self.original_data.clone()
        d.x = x
        d.edge_index = edge_index
        
        d.edge_attr = getattr(self.original_data, "edge_attr", None)
        d.atom_megnet_embed = getattr(self.original_data, "atom_megnet_embed", None)
        d.cart_dist = getattr(self.original_data, "cart_dist", None)
        d.batch = getattr(self.original_data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        return self.base_model(d)