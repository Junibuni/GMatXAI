import os
import argparse
import hashlib
import pandas as pd
import yaml
import itertools
from itertools import product
from pathlib import Path
from copy import deepcopy

from src.data.run_single import run_single_experiment
from src.utils.config import load_config


def hash_config(cfg_dict):
    hash_input = str(cfg_dict)
    return hashlib.md5(hash_input.encode()).hexdigest()[:6]

def is_sweep_config(config: dict) -> bool:
    def contains_list(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k in ["explain_material_ids", "target"]:
                    continue
                if isinstance(v, list):
                    if any(isinstance(i, dict) for i in v):
                        if contains_list_in_list_of_dicts(v):
                            return True
                    else:
                        return True
                if isinstance(v, dict) and contains_list(v):
                    return True
        return False

    def contains_list_in_list_of_dicts(lst):
        for item in lst:
            if isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, list):
                        return True
        return False

    return contains_list(config)

def expand_optimizer_configs(optimizers):
    expanded_optimizers = []
    for opt in optimizers:
        sweep_keys = [k for k, v in opt.items() if isinstance(v, list)]
        if not sweep_keys:
            expanded_optimizers.append(opt)
            continue
        sweep_vals = [opt[k] for k in sweep_keys]
        for combo in itertools.product(*sweep_vals):
            new_opt = deepcopy(opt)
            for k, v in zip(sweep_keys, combo):
                new_opt[k] = v
            expanded_optimizers.append(new_opt)
    return expanded_optimizers

def generate_sweep_combinations(config):
    base_config = deepcopy(config)
    sweep_params = {}

    def extract_sweep_params(d, path=()):
        for k, v in d.items():
            new_path = path + (k,)
            if k in ["explain_material_ids", "target"]:
                continue
            if k == "optimizer" and path == ("training",):
                expanded = expand_optimizer_configs(v)
                sweep_params[new_path] = expanded
                continue
            if isinstance(v, list) and not any(isinstance(i, dict) for i in v):
                sweep_params[new_path] = v
            elif isinstance(v, dict):
                extract_sweep_params(v, new_path)

    extract_sweep_params(config)

    if not sweep_params:
        return [config]

    keys, values = zip(*sweep_params.items())
    combinations = []

    for prod in itertools.product(*values):
        new_config = deepcopy(base_config)
        for key_path, val in zip(keys, prod):
            d = new_config
            for k in key_path[:-1]:
                d = d[k]
            d[key_path[-1]] = val
        combinations.append(new_config)

    return combinations


def run_sweep(config_path):
    base_config = load_config(config_path).to_dict()
    sweep_name = Path(config_path).stem
    sweep_dir = Path("outputs") / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_configs = generate_sweep_combinations(base_config)

    print(f"Running sweep: {sweep_name} with {len(sweep_configs)} configs")

    sweep_results = []
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

    if is_sweep_config(cfg):
        run_sweep(args.config)
    else:
        run_single_experiment(args.config)

if __name__ == "__main__":
    main()
