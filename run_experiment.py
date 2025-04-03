import argparse
import torch
import os

from src.utils.config import load_config
from src.utils.io import save_model
from src.models.cgcnn import CGCNN
from src.data.loader import get_loaders
from src.train.trainer import Trainer
from src.utils.optim import get_optimizer, get_scheduler

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
    
    print("\nLoad Dataset")
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

    optimizer_cfg = cfg.training.optimizer
    scheduler_cfg = cfg.training.scheduler

    optimizer = get_optimizer(
        optim_type=optimizer_cfg.name,
        model_parameters=model.parameters(),
        **optimizer_cfg
    )

    scheduler = get_scheduler(
        scheduler_type=scheduler_cfg.name,
        optimizer=optimizer,
        **scheduler_cfg
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.training.device
    )

    print("\nStart Training")
    best_model, train_losses, val_maes = trainer.train(num_epochs=cfg.training.epochs)

    save_model(best_model, cfg.experiment.save_dir, cfg.experiment.model_name)

    model.load_state_dict(best_model)
    trainer.test(test_loader, metric="mae")
    trainer.test(test_loader, metric="rmse")
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_maes, label='val loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

if __name__ == "__main__":
    main()
