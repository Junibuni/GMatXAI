from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.utils.metrics import mae, mse, rmse

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        log_dir,
        device="cpu",
        loss_fn=nn.L1Loss(),
        scheduler=None,
    ):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.lr_history = []
        self.scheduler_step_per_batch = self.should_step_per_batch()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loader = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False, dynamic_ncols=True)

        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch)
            loss = self.loss_fn(pred, batch.y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler and self.scheduler_step_per_batch:
                self.scheduler.step()

            total_loss += loss.item() * batch.num_graphs
            loader.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader.dataset)

    def validate(self):
        total_loss = 0
        self.model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                batch = batch.to(self.device)
                pred = self.model(batch)
                preds.append(pred.cpu())
                targets.append(batch.y.cpu())
                
                loss = self.loss_fn(pred, batch.y)
                total_loss += loss.item() * batch.num_graphs

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        avg_loss = total_loss / len(self.val_loader.dataset)
        val_mae = mae(preds, targets)
        
        return avg_loss, val_mae

    def train(self, num_epochs=30, max_prints=20):
        best_val_mae = float("inf")
        best_model = None
        step = (num_epochs + max_prints - 1) // max_prints

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_mae = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae)
            current_lr = self.get_current_lr()
            self.lr_history.append(current_lr)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Metric/val_mae", val_mae, epoch)
            self.writer.add_scalar("LR", current_lr, epoch)
            
            # Gradient norm 로깅
            total_norm_sq = torch.tensor(0.0, device=self.device)
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_sq += param_norm ** 2
            total_norm = torch.sqrt(total_norm_sq).item()
            self.writer.add_scalar("Gradients/global_norm", total_norm, epoch)

            if epoch == 1 or epoch % step == 0 or epoch == num_epochs:
                print(f"[Epoch {epoch}] LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model = self.model.state_dict()

            if self.scheduler:
                if self.scheduler_step_per_batch:
                    pass
                elif "ReduceLROnPlateau" in self.scheduler.__class__.__name__:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

        print(f"Best Validation MAE: {best_val_mae:.4f}")
        return best_model
        
    def test(self, metric='mae'):
        self.model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing", leave=False):
                batch = batch.to(self.device)
                pred = self.model(batch)
                preds.append(pred.cpu())
                targets.append(batch.y.cpu())

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        if metric == "mae":
            score = mae(preds, targets)
        elif metric == "rmse":
            score = rmse(preds, targets)
        elif metric == "mse":
            score = mse(preds, targets)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        print(f"Test {metric}: {score:.4f}")
        return score
    
    def get_current_lr(self):
        if self.scheduler and hasattr(self.scheduler, "get_last_lr"):
            return self.scheduler.get_last_lr()[0]
        return self.optimizer.param_groups[0]["lr"]
    
    def export_logs_to_csv(self, save_path):
        import pandas as pd
        df = pd.DataFrame({
            "epoch": list(range(1, len(self.train_losses) + 1)),
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "val_mae": self.val_maes,
            "lr": self.lr_history
        })
        df.to_csv(save_path, index=False)
        print(f"Logs exported to {save_path}")
        
    def should_step_per_batch(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            return True
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.CyclicLR):
            return True
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.LambdaLR):
            return True
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR) and hasattr(self.scheduler, 'T_max'):
            return True
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return False  # This requires val_loss input and steps per epoch
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
            return False
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.MultiStepLR):
            return False
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ExponentialLR):
            return False
        else:
            return False  # Default to epoch-based if unknown scheduler