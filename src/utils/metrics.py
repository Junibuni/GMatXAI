import torch

def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def mse(pred, target):
    return torch.mean((pred - target) ** 2).item()

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

def r2_score(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return (1 - ss_res / ss_tot).item()

def mape(pred, target):
    epsilon = 1e-8
    return torch.mean(torch.abs((target - pred) / (target + epsilon))).item() * 100
