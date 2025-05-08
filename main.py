import os
import argparse
import hashlib
import pandas as pd
import yaml
import itertools
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path
from copy import deepcopy

from src.data.run_single import run_single_experiment
from src.utils.config import load_config
from src.utils.analysis.pcp import load_dataframe, pcp
from src.utils.analysis.importance import analyze_param_importance


def hash_config(cfg_dict):
    hash_input = str(cfg_dict)
    return hashlib.md5(hash_input.encode()).hexdigest()[:6]

def run_analysis(config_path, to_track):
    sweep_name = Path(config_path).stem
    sweep_dir = Path("outputs") / sweep_name
    experiment_folders = [p for p in sweep_dir.iterdir() if p.is_dir() and p.name.startswith("test")]
    
    sweep_results = []
    print(f"Processing {len(experiment_folders)} folders in {sweep_dir}")
    for folder_pth in experiment_folders:
        tag = folder_pth.stem
        log_folder_path = folder_pth / "logs"
        cfg_dict = load_config(log_folder_path / "config.yaml")
        flat_params = flatten_config(cfg_dict, keys_to_track=to_track)
        
        log_path = log_folder_path / "log.csv"
        try:
            df = pd.read_csv(log_path)
            best_row = {"val_mae": df["val_mae"].min()}
        except Exception as e:
            best_row = {"error": str(e)}
            
        result = {"tag": tag, **flat_params, **best_row}
        sweep_results.append(result)

    result_df = pd.DataFrame(sweep_results)
    
    # Parallel Plot
    data, labels = load_dataframe(result_df)
    # If metric=loss --> flip_metric=True
    fig = pcp(data, labels, alpha=0.8, flip_metric=True)
    plt.savefig(sweep_dir / f"{sweep_name}_parallel_plot.svg", format="svg", facecolor="white", bbox_inches="tight")
    plt.close()
    # Feature Importance
    feature_importance_df = analyze_param_importance(result_df)
    feature_importance_df.to_csv(sweep_dir / f"{sweep_name}_feature_importance.csv", index=False)
    
    result_df.to_csv(sweep_dir / f"{sweep_name}.csv", index=False)
    print(f"\nSweep analysis complete. Results saved to {sweep_dir / f'{sweep_name}.csv'}")
    
def get_sweep_keys(config: dict) -> list:
    sweep_keys = set()

    def find_lists(d, path=()):
        for k, v in d.items():
            if k in ["explain_material_ids", "target"]:
                continue
            current_path = path + (k,)
            if isinstance(v, list):
                if all(isinstance(i, dict) for i in v):
                    if current_path == ("training", "optimizer"):
                        sweep_keys.add("optimizer")
                    elif contains_list_in_list_of_dicts(v):
                        sweep_keys.add(k)
                else:
                    sweep_keys.add(k)
            elif isinstance(v, dict):
                find_lists(v, current_path)

    def contains_list_in_list_of_dicts(lst):
        for item in lst:
            if isinstance(item, dict):
                for val in item.values():
                    if isinstance(val, list):
                        return True
        return False

    find_lists(config)
    return sorted(sweep_keys)

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

def expand_scheduler_configs(schedulers):
    expanded_schedulers = []
    for sched in schedulers:
        sweep_keys = [k for k, v in sched.items() if isinstance(v, list)]
        sweep_vals = [sched[k] for k in sweep_keys]
        for combo in itertools.product(*sweep_vals):
            new_sched = deepcopy(sched)
            for k, v in zip(sweep_keys, combo):
                new_sched[k] = v
            expanded_schedulers.append(new_sched)
    return expanded_schedulers

def generate_sweep_combinations(config):
    base_config = deepcopy(config)
    sweep_params = {}

    def extract_sweep_params(d, path=()):
        for k, v in d.items():
            new_path = path + (k,)
            if k in ["explain_material_ids", "target"]:
                continue
            if k == "optimizer" and path == ("training",):
                if isinstance(v, dict):
                    optim_list = [v]
                else:
                    optim_list = v
                expanded = expand_optimizer_configs(optim_list)
                sweep_params[new_path] = expanded
                continue

            if k == "scheduler" and path == ("training",):
                if isinstance(v, dict):
                    sched_list = [v]
                else:
                    sched_list = v
                expanded = expand_scheduler_configs(sched_list)
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

def flatten_config(cfg_dict, keys_to_track=None):
    flat = {}

    def extract(d, parent=""):
        for k, v in d.items():
            full_key = f"{parent}.{k}" if parent else k
            if isinstance(v, dict):
                extract(v, full_key)
            elif isinstance(v, (list, tuple)) and all(isinstance(i, (float, int)) for i in v):
                for i, vi in enumerate(v):
                    flat[f"{full_key}_{i}"] = vi
            else:
                flat[full_key] = v

    extract(cfg_dict)

    if keys_to_track:
        filtered = {}
        for k, v in flat.items():
            for key in keys_to_track:
                if key in k:
                    clean_key = (
                        k.replace("training.optimizer.", "optimizer_")
                         .replace("training.", "")
                         .replace("model.", "")
                         .replace("data.", "")
                    )
                    filtered[clean_key] = v
        return filtered

    return flat

def run_sweep(config_path, to_track):
    base_config = load_config(config_path).to_dict()
    sweep_name = Path(config_path).stem
    sweep_dir = Path("outputs") / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_configs = generate_sweep_combinations(base_config)

    print(f"Running sweep: {sweep_name} with {len(sweep_configs)} configs")

    sweep_results = []
    for i, cfg_dict in enumerate(sweep_configs, start=1):
        tag = f"test_{i:03d}_{hash_config(cfg_dict)}"
        print(f"\n=============== Start {tag} ({i}/{len(sweep_configs)}) ===============")
        
        flat_params = flatten_config(cfg_dict, keys_to_track=to_track)
        cfg_path = sweep_dir / f"{tag}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg_dict, f)

        print(f"\nHyperparams: {flat_params}")
        tag_full, log_path = run_single_experiment(str(cfg_path), tag_override=f"{sweep_name}/{tag}")

        try:
            df = pd.read_csv(log_path)
            best_row = {"val_mae": df["val_mae"].min()}
        except Exception as e:
            best_row = {"error": str(e)}

        result = {"tag": tag, **flat_params, **best_row}
        sweep_results.append(result)

    print(f"\n=====================================")
    result_df = pd.DataFrame(sweep_results)
    
    # Parallel Plot
    data, labels = load_dataframe(result_df)
    fig = pcp(data, labels, alpha=0.8)
    plt.savefig(sweep_dir / f"{sweep_name}_parallel_plot.svg", format="svg", facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Save Parallel Plot")
    # Feature Importance
    feature_importance_df = analyze_param_importance(result_df)
    feature_importance_df.to_csv(sweep_dir / f"{sweep_name}_feature_importance.csv", index=False)
    print(f"Save Feature Importance")
    
    result_df.to_csv(sweep_dir / f"{sweep_name}.csv", index=False)
    print(f"\nSweep complete. Results saved to: {sweep_dir / f'{sweep_name}.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-A", "--analyze_results", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sweep_keys = get_sweep_keys(cfg)
    
    if args.analyze_results:
        print("Running analysis...")
        if not sweep_keys:
            print("Not a sweep experiment")
            return
        run_analysis(args.config, sweep_keys)
        return

    if sweep_keys:
        print("Sweep keys:", sweep_keys)
        run_sweep(args.config, sweep_keys)
    else:
        run_single_experiment(args.config)

if __name__ == "__main__":
    main()
