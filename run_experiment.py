import argparse
import torch
import os

from src.utils.config import load_config
from src.utils.io import save_model
from src.models.cgcnn import CGCNN
from src.data.loader import get_loaders
from src.train.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("Loaded config:")
    for section in ["experiment", "data", "model", "training"]:
        print(f"[{section}]")
        for k, v in getattr(cfg, section).items():
            print(f"  {k}: {v}")

    train_loader, val_loader, test_loader = get_loaders(
        data_dir=cfg.data.data_dir,
        target=cfg.data.target,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed
    )

    model = CGCNN(
        node_input_dim=cfg.model.node_input_dim,
        edge_input_dim=cfg.model.edge_input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        output_dim=cfg.model.output_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=cfg.training.device
    )

    best_model = trainer.train(num_epochs=cfg.training.epochs)

    save_model(best_model, cfg.experiment.save_dir, cfg.experiment.model_name)

    model.load_state_dict(best_model)
    trainer.test(test_loader, metric="mae")
    trainer.test(test_loader, metric="rmse")

if __name__ == "__main__":
    main()
