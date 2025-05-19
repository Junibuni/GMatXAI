import torch
import torch.nn as nn

class OutlierAwareL1Loss(nn.Module):
    def __init__(self, threshold=0.5, outlier_weight=3.0, reduction='mean'):
        super().__init__()
        self.threshold = threshold
        self.outlier_weight = outlier_weight
        self.reduction = reduction

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        weights = torch.where(error > self.threshold,
                              torch.tensor(self.outlier_weight, device=error.device),
                              torch.tensor(1.0, device=error.device))
        loss = weights * error
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class LogCoshLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss = torch.log(torch.cosh(pred - target))
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def get_loss_function(cfg):
    """
    ['l1', 'l2', 'mse', 'huber', 'logcosh', 'outlier_l1']
    """
    if not isinstance(cfg, dict) or "type" not in cfg:
        raise ValueError("Loss config must be a dict with a 'type' key.")

    loss_type = cfg.type.lower()
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    
    name = loss_type.lower()
    if name == 'l1':
        return nn.L1Loss(**kwargs)
    elif name in ('l2', 'mse'):
        return nn.MSELoss(**kwargs)
    elif name == 'huber':
        return nn.HuberLoss(delta=kwargs.get("delta", 0.3), reduction=kwargs.get("reduction", "mean"))
    elif name == 'logcosh':
        return LogCoshLoss(reduction=kwargs.get("reduction", "mean"))
    elif name in ('outlier_l1', 'outlier'):
        return OutlierAwareL1Loss(
            threshold=kwargs.get("threshold", 0.5),
            outlier_weight=kwargs.get("outlier_weight", 3.0),
            reduction=kwargs.get("reduction", "mean")
        )
    else:
        raise ValueError(f"Unsupported loss: {name}")
