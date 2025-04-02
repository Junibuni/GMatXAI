from tqdm import tqdm

import torch
import torch.nn as nn

from src.utils.metrics import mae

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device="cpu",
        loss_fn=nn.L1Loss(),
        scheduler=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.scheduler = scheduler

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

            total_loss += loss.item() * batch.num_graphs
            loader.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                batch = batch.to(self.device)
                pred = self.model(batch)
                preds.append(pred.cpu())
                targets.append(batch.y.cpu())

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        return mae(preds, targets)

    def train(self, num_epochs=30, max_prints=20):
        best_val_mae = float("inf")
        best_model = None
        step = (num_epochs + max_prints - 1) // max_prints

        train_losses = []
        val_maes = []

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_mae = self.validate()

            train_losses.append(train_loss)
            val_maes.append(val_mae)

            if epoch == 1 or epoch % step == 0 or epoch == num_epochs:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model = self.model.state_dict()

            if self.scheduler:
                self.scheduler.step()

        print(f"Best Validation MAE: {best_val_mae:.4f}")
        return best_model, train_losses, val_maes
