import os
import yaml
import time

import torch.nn as nn
import matplotlib.pyplot as plt

from src.models.model_registry import get_model
from src.data.loader import get_loaders
from src.train.trainer import Trainer
from src.utils.io import save_model
from src.utils.seed import set_seed
from src.utils.config import load_config
from src.utils.loss import get_loss_function
from src.utils.data import sample_explanation_data
from src.utils.analysis.parity_plot import plot_parity
from src.utils.optim import get_optimizer, get_scheduler
from src.xai.wrappers import CGCNNWrapper
from src.xai.batch_explainer import explain_batch


def create_output_dirs(tag, root="outputs"):
    log_dir = os.path.join(root, tag, "logs")
    ckpt_dir = os.path.join(root, tag, "checkpoints")
    explain_dir = os.path.join(root, tag, "explain")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(explain_dir, exist_ok=True)
    return log_dir, ckpt_dir, explain_dir


def run_single_experiment(config_path: str, tag_override: str = None):
    cfg = load_config(config_path)

    tag = tag_override or os.path.splitext(os.path.basename(config_path))[0]
    log_dir, ckpt_dir, explain_dir = create_output_dirs(tag=tag)

    set_seed(cfg.data.seed)

    print(f"\nLoad Dataset: {cfg.data.target}")
    norm = True
    mean, std = -0.9633, 1.0722
    if norm:
        print(f"Normalize data with mean({mean}), std({std})")
    ## Legacy
    # train_loader, val_loader, test_loader = get_loaders(
    #     data_dir=cfg.data.data_dir,
    #     target=cfg.data.target,
    #     batch_size=cfg.data.batch_size,
    #     num_workers=cfg.data.num_workers,
    #     train_ratio=cfg.data.train_ratio,
    #     val_ratio=cfg.data.val_ratio,
    #     onehot=cfg.data.onehot,
    #     jitter_std=cfg.data.jitter_std,
    #     seed=cfg.data.seed,
    #     norm=norm,
    #     mean=mean,
    #     std=std
    # )
    train_loader, val_loader, test_loader = get_loaders(
        data_dir=cfg.data.data_dir,
        target=cfg.data.target,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        onehot=cfg.data.onehot,
        jitter_std=cfg.data.jitter_std,
        seed=cfg.data.seed,
        norm=norm,
        mean=mean,
        std=std
    )

    if cfg.model.output_dim:
        assert len(cfg.data.target) == cfg.model.output_dim,\
            f"The number of targets({len(cfg.data.target)}) does not match config.model.output_dim({cfg.model.output_dim})"
    
    print(f"\nLoad Model ({cfg.experiment.model_name}) with Hyperparamters:\n\t{cfg.model}")
    model = get_model(cfg.experiment.model_name, config=cfg.model)

    print(f"\nOptimizer Config: \n\t{cfg.training.optimizer}")
    optimizer = get_optimizer(
        optim_type=cfg.training.optimizer.name,
        model_parameters=model.parameters(),
        lr=cfg.training.lr,
        **cfg.training.optimizer
    )
    
    cfg.training.scheduler.epochs = cfg.training.epochs
    cfg.training.scheduler.steps_per_epoch = len(train_loader)
    print(f"\nScheduler Config: \n\t{cfg.training.scheduler}")
    scheduler = get_scheduler(
        scheduler_type=cfg.training.scheduler.name,
        optimizer=optimizer,
        **cfg.training.scheduler
    )

    print(f"\nLoss Funciton Config: \n\t{cfg.training.loss_fn}")
    loss_fn = get_loss_function(cfg.training.loss_fn)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.training.device,
        log_dir=log_dir,
        loss_fn=loss_fn
    )

    print(f"\nStart Training {tag} With \n\t{ {k: cfg.training.get(k) for k in ['device', 'epochs', 'lr']} }", end="\n\n")
    start_time = time.time()
    best_model = trainer.train(num_epochs=cfg.training.epochs)
    end_time = time.time()
    print(f"Elapsed Time: {end_time-start_time:.2f} seconds")
    
    save_model(best_model, ckpt_dir, cfg.experiment.model_name)
    trainer.export_logs_to_csv(os.path.join(log_dir, "log.csv"))

    print("\nFinal Evaluation")
    trainer.load_model(best_model)
    trainer.test()

    print("\nSave Loss Curve")
    plt.figure()
    plt.plot(trainer.train_losses, label="train loss")
    plt.plot(trainer.val_maes, label='val loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(log_dir, "loss.png"))
    plt.close()

    print("\nRun XAI")
    material_ids = getattr(cfg.experiment, "explain_material_ids", None)
    selected_data = sample_explanation_data(test_loader.dataset, material_ids, k=3)
    explain_batch(
        model=CGCNNWrapper(model),
        data_list=selected_data,
        save_dir=explain_dir,
        epochs=100,
        edge_threshold=0.3,
        scale_node_size=True,
        algorithm="gnn_explainer"
    )

    print("\nPlot Parity")
    plot_parity(model, test_loader, log_dir, device=cfg.training.device)
    
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg.to_dict(), f)

    print(f"\nFinished: {tag}")
    return tag, os.path.join(log_dir, "log.csv")
