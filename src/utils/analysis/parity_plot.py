import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import os

def plot_parity(
    model, 
    test_loader, 
    save_path: str, 
    device: str = "cpu"
):
    model.to(device)
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.append(pred.view(-1).cpu())
            trues.append(batch.y.view(-1).cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    r2 = r2_score(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = root_mean_squared_error(trues, preds)

    plt.figure(figsize=(6, 6))
    plt.scatter(trues, preds, alpha=0.6, color='black')
    min_val = min(trues.min(), preds.min())
    max_val = max(trues.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', label="y = x")

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.axis("equal")

    plt.text(
        0.05, 0.95, 
        f"$R^2$: {r2:.3f}\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )

    plt.tight_layout()

    plt.savefig(os.path.join(save_path, "parity_plot.png"), dpi=300)
    plt.close()
