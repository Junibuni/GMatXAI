import torch

def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()
