import os
import yaml
import torch
from pathlib import Path

from src.models.model_registry import get_model
from src.data.loader import get_loaders
from src.train.trainer import Trainer
from src.utils.config import load_config
from src.utils.loss import get_loss_function
from src.utils.seed import set_seed
from src.utils.analysis.parity_plot import plot_parity


def create_test_output_dirs(tag, root="outputs"):
    log_dir = os.path.join(root, tag, "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def parse_yaml_and_model_path(yaml_path: str):
    cfg = load_config(yaml_path)

    root_dir = os.path.dirname(yaml_path)
    model_name = cfg.experiment.model_name
    model_path = os.path.join(root_dir, Path(yaml_path).stem, "checkpoints", f"{model_name}_best.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at expected location: {model_path}")

    print(f"Processing {model_path}")

    return cfg, model_path, root_dir


def run_test(yaml_path: str):
    cfg, model_path, base_dir = parse_yaml_and_model_path(yaml_path)

    tag = f"test_{os.path.splitext(os.path.basename(yaml_path))[0]}"
    log_dir = create_test_output_dirs(tag)

    set_seed(cfg.data.seed)

    print(f"\n[TEST] Load Dataset: {cfg.data.target}")
    norm = cfg.data.isNorm
    mean, std = -0.9633, 1.0722
    if norm:
        print(f"[TEST] Normalize with mean({mean}), std({std})")

    _, _, test_loader = get_loaders(
        data_dir=cfg.data.data_dir,
        target=cfg.data.target,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        radius=cfg.data.radius,
        seed=cfg.data.seed,
        dataset_name=cfg.data.dataset,
        max_neighbors=25,
        mean=mean,
        std=std,
        norm=norm
    )

    print(f"\n[TEST] Load Model: {cfg.experiment.model_name}")
    model = get_model(cfg.experiment.model_name, config=cfg.model)
    model.load_state_dict(torch.load(model_path, map_location=cfg.training.device))
    model.to(cfg.training.device)

    print(f"\n[TEST] Load Loss Function: {cfg.training.loss_fn}")
    loss_fn = get_loss_function(cfg.training.loss_fn)

    trainer = Trainer(
        model=model,
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
        optimizer=None,
        scheduler=None,
        device=cfg.training.device,
        log_dir=log_dir,
        loss_fn=loss_fn
    )

    print("\n[TEST] Start Evaluation")
    trainer.test()

    print("\n[TEST] Plot Parity")
    plot_parity(model, test_loader, log_dir, device=cfg.training.device, isNorm=cfg.data.isNorm)

    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg.to_dict(), f)

    print(f"\n[TEST] Finished: {tag}")
    return tag