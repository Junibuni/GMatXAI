import os
import argparse
import hashlib
import pandas as pd
import itertools
from pathlib import Path
from copy import deepcopy
import yaml

from src.data.run_single import run_single_experiment
from src.utils.config import load_config


def hash_config(cfg_dict):
    hash_input = str(cfg_dict)
    return hashlib.md5(hash_input.encode()).hexdigest()[:6]


def create_sweep_combinations(config):
    keys, values = [], []

    for section, params in config.items():
        if not isinstance(params, dict):
            continue
        for key, val in params.items():
            if isinstance(val, list) and val and not isinstance(val[0], dict):
                keys.append((section, key))
                values.append(val)

    combos = list(itertools.product(*values)) if values else [()]
    configs = []

    for combo in combos:
        cfg_copy = deepcopy(config)
        for (section, key), value in zip(keys, combo):
            if section == "data" and key == "target":
                value = [value]
            cfg_copy[section][key] = value
        configs.append(cfg_copy)

    return configs


def expand_optimizer_options(optimizer_list):
    from itertools import product
    expanded = []

    for opt in optimizer_list:
        sweep_keys = [k for k, v in opt.items() if isinstance(v, list)]
        if not sweep_keys:
            expanded.append(opt)
            continue

        sweep_values = [opt[k] for k in sweep_keys]
        for values in product(*sweep_values):
            new_opt = deepcopy(opt)
            for k, v in zip(sweep_keys, values):
                new_opt[k] = v
            expanded.append(new_opt)

    return expanded


def run_sweep(config_path):
    base_config = load_config(config_path).to_dict()
    sweep_name = Path(config_path).stem
    sweep_dir = Path("outputs") / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(base_config.get("training", {}).get("optimizer"), list) and isinstance(base_config["training"]["optimizer"][0], dict):
        base_config["training"]["optimizer"] = expand_optimizer_options(base_config["training"]["optimizer"])

    sweep_configs = create_sweep_combinations(base_config)
    sweep_results = []

    print(f"Running sweep: {sweep_name} with {len(sweep_configs)} configs")

    for i, cfg_dict in enumerate(sweep_configs, start=1):
        tag = f"test_{i:03d}_{hash_config(cfg_dict)}"
        cfg_path = sweep_dir / f"{tag}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg_dict, f)

        tag_full, log_path = run_single_experiment(str(cfg_path), tag_override=f"{sweep_name}/{tag}")

        try:
            df = pd.read_csv(log_path)
            best_idx = df["val_mae"].idxmin()
            best_row = df.iloc[best_idx].to_dict()
        except Exception as e:
            best_row = {"error": str(e)}

        best_row.update({"tag": tag})
        sweep_results.append(best_row)

    result_df = pd.DataFrame(sweep_results)
    result_df.to_csv(sweep_dir / f"{sweep_name}.csv", index=False)
    print(f"Sweep complete. Results saved to {sweep_dir / f'{sweep_name}.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg.test_type == "single":
        run_single_experiment(args.config)
    elif cfg.test_type == "sweep":
        run_sweep(args.config)
    else:
        raise ValueError(f"Unsupported test_type: {cfg.test_type}")


if __name__ == "__main__":
    main()
