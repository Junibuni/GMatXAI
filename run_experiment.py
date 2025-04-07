import os
import yaml
import random
import argparse
from datetime import datetime

import torch

from src.utils.config import load_config
from src.utils.io import save_model
from src.models.cgcnn import CGCNN
from src.data.loader import get_loaders
from src.train.trainer import Trainer
from src.utils.optim import get_optimizer, get_scheduler
from src.utils.seed import set_seed
from src.utils.data import sample_explanation_data
from src.xai.wrappers import CGCNNWrapper
from src.xai.batch_explainer import explain_batch

def create_log_dir(config):
    date = datetime.now().strftime("%Y_%m_%d")
    model = config.experiment.model_name
    target = config.data.target
    lr = config.training.optimizer.lr
    bs = config.data.batch_size

    name = f"{date}_{model}_{'_'.join(target)}_lr{lr}_bs{bs}"
    log_dir = os.path.join("outputs", "logs", name)
    ckpt_dir = os.path.join("outputs", "checkpoints", name)
    explain_dir = os.path.join("outputs", "explain", name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(explain_dir, exist_ok=True)
    return log_dir, ckpt_dir, explain_dir

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
    
    log_dir, ckpt_dir, explain_dir = create_log_dir(cfg)
    
    set_seed(cfg.data.seed)

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
        #output_dim=cfg.model.output_dim,
        output_dim=len(cfg.data.target)
    )

    optimizer_cfg = cfg.training.optimizer
    scheduler_cfg = cfg.training.scheduler

    optimizer = get_optimizer(
        optim_type=optimizer_cfg.name,
        model_parameters=model.parameters(),
        **optimizer_cfg
    )

    # StepLR, CosineAnnealingLR
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
        device=cfg.training.device,
        log_dir=log_dir
    )

    print("\nStart Training")
    best_model = trainer.train(num_epochs=cfg.training.epochs)

    save_model(best_model, ckpt_dir, cfg.experiment.model_name)
    trainer.export_logs_to_csv(f"{log_dir}/log.csv")

    print("\nTest Model")
    model.load_state_dict(best_model)
    trainer.test(test_loader, metric="mae")
    trainer.test(test_loader, metric="rmse")

    print("\nSave Loss Plot")    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(trainer.train_losses, label="train loss")
    plt.plot(trainer.val_maes, label='val loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(log_dir, "loss.png"))

    # Explainer
    print("\nXAI Explainer")  
    material_ids = getattr(cfg.experiment, "explain_material_ids", None)
    selected_data = sample_explanation_data(test_loader.dataset, material_ids, k=3)
    
    explain_batch(
        model=CGCNNWrapper(model),
        data_list=selected_data,
        save_dir=explain_dir,
        device=cfg.training.device,
        epochs=100,
        edge_threshold=0.3,
        scale_node_size=True,
        algorithm="gnn_explainer"  # "gnn_explainer", 
    )
    
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg.to_dict(), f)

if __name__ == "__main__":
    main()
