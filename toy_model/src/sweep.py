"""
Sweep script for running multiple experiments with different configurations.
Supports both grid search and random search over data and optimizer parameters.
"""

import argparse
import itertools
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List

import yaml

from data.builder import build_dataset_with_functor, save_dataset
from train import train


def load_sweep_config(config_path: str) -> Dict:
    """Load sweep configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_grid_combinations(sweep_params: Dict[str, List]) -> List[Dict]:
    """Generate all combinations for grid search."""
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def generate_random_combinations(sweep_params: Dict[str, List], n_samples: int) -> List[Dict]:
    """Generate random combinations for random search."""
    combinations = []
    for _ in range(n_samples):
        combo = {k: random.choice(v) for k, v in sweep_params.items()}
        combinations.append(combo)
    return combinations


def run_single_experiment(
    data_params: Dict,
    optimizer_params: Dict,
    model_params: Dict,
    fixed_params: Dict,
    base_data_dir: str,
    base_runs_dir: str,
    wandb_project: str,
    experiment_id: str,
    dry_run: bool = False,
):
    """Run a single experiment with given parameters."""
    
    # Calculate sub_size from num_entities (minimum 1)
    num_entities = data_params["num_entities"]
    sub_size = max(1, num_entities // 2)
    
    # Get include_f_inverse from data_params or fixed_params
    include_f_inverse = data_params.get("include_f_inverse", fixed_params.get("include_f_inverse", False))
    
    # Create unique data directory name
    data_name = (
        f"e{data_params['num_entities']}_"
        f"r{data_params['num_relations']}_"
        f"ao{data_params['atomic_ood_ratio']}_"
        f"co{data_params['compositional_ood_ratio']}_"
        f"an{data_params['analogical_ood_ratio']}_"
        f"inv{int(include_f_inverse)}"
    )
    data_dir = os.path.join(base_data_dir, data_name)
    
    # Resolve model params (model_params overrides fixed_params)
    d_model = model_params.get("d_model", fixed_params.get("d_model", 128))
    n_layer = model_params.get("n_layer", fixed_params.get("n_layer", 1))
    n_head = model_params.get("n_head", fixed_params.get("n_head", 1))
    dropout = model_params.get("dropout", fixed_params.get("dropout", 0.0))
    max_len = model_params.get("max_len", fixed_params.get("max_len", 64))
    train_seed = model_params.get("seed", fixed_params.get("seed", 42))
    
    # Create unique run directory
    model_tag = f"dm{d_model}_nl{n_layer}_nh{n_head}_s{train_seed}"
    run_name = (
        f"{experiment_id}_"
        f"{model_tag}_"
        f"lr{optimizer_params['lr']}_"
        f"wd{optimizer_params['weight_decay']}_"
        f"bs{optimizer_params['batch_size']}"
    )
    save_dir = os.path.join(base_runs_dir, run_name)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {run_name}")
    print(f"Data params: {data_params}")
    print(f"Optimizer params: {optimizer_params}")
    print(f"Model params: {model_params}")
    print(f"{'='*60}")
    
    if dry_run:
        print("[DRY RUN] Skipping actual execution")
        return
    
    # Generate dataset if not exists
    if not os.path.exists(os.path.join(data_dir, "train.json")):
        print(f"Generating dataset: {data_dir}")
        res = build_dataset_with_functor(
            num_entities=data_params["num_entities"],
            num_relations=data_params["num_relations"],
            sub_size=sub_size,
            atomic_ood_ratio=data_params["atomic_ood_ratio"],
            compositional_ood_ratio=data_params["compositional_ood_ratio"],
            analogical_ood_ratio=data_params["analogical_ood_ratio"],
            seed=fixed_params.get("data_seed", fixed_params.get("seed", 42)),
            include_f_inverse=include_f_inverse,
            duplicate_relation=fixed_params.get("duplicate_relation", False),
        )
        
        (entities, relations,
         id_atomic_facts, ood_atomic_facts,
         id_compositional_facts, near_ood_compositional_facts, far_ood_compositional_facts,
         id_analogical_facts, ood_analogical_facts) = res
        
        save_dataset(
            data_dir,
            entities, relations,
            id_atomic_facts, ood_atomic_facts,
            id_compositional_facts, near_ood_compositional_facts, far_ood_compositional_facts,
            id_analogical_facts, ood_analogical_facts,
        )
    else:
        print(f"Dataset already exists: {data_dir}")
    
    # Build training config
    config = {
        # Data params (for WandB logging)
        "num_entities": data_params["num_entities"],
        "num_relations": data_params["num_relations"],
        "sub_size": sub_size,
        "atomic_ood_ratio": data_params["atomic_ood_ratio"],
        "compositional_ood_ratio": data_params["compositional_ood_ratio"],
        "analogical_ood_ratio": data_params["analogical_ood_ratio"],
        "include_f_inverse": include_f_inverse,
        "duplicate_relation": fixed_params.get("duplicate_relation", False),
        
        # Paths
        "data_dir": data_dir,
        "save_dir": save_dir,
        
        # Model params
        "d_model": d_model,
        "n_layer": n_layer,
        "n_head": n_head,
        "dropout": dropout,
        "max_len": max_len,
        
        # Optimizer params
        "lr": optimizer_params["lr"],
        "weight_decay": optimizer_params["weight_decay"],
        "batch_size": optimizer_params["batch_size"],
        
        # Training params
        "epochs": fixed_params.get("epochs", 100),
        "warmup_steps": fixed_params.get("warmup_steps", 0),
        "use_amp": fixed_params.get("use_amp", True),
        
        # Logging
        "use_wandb": True,
        "project": wandb_project,
        "run_name": run_name,
        "log_every": fixed_params.get("log_every", 50),
        "eval_every": fixed_params.get("eval_every", 10),
        "save_every": fixed_params.get("save_every", 100),
        "num_workers": fixed_params.get("num_workers", 4),
        "seed": train_seed,
    }
    
    # Run training
    try:
        train(config)
    finally:
        # Ensure wandb is properly finished
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run sweep experiments")
    parser.add_argument("--config", type=str, default="configs/sweep_config.yaml",
                        help="Path to sweep config file")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "random"],
                        help="Sweep mode: grid or random")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of samples for random search")
    parser.add_argument("--data_dir", type=str, default="/app/data/sweep",
                        help="Base directory for generated datasets")
    parser.add_argument("--runs_dir", type=str, default="/app/runs/sweep",
                        help="Base directory for experiment runs")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name")
    parser.add_argument("--sweep_data", action="store_true",
                        help="Sweep over data parameters")
    parser.add_argument("--sweep_optimizer", action="store_true",
                        help="Sweep over optimizer parameters")
    parser.add_argument("--sweep_model", action="store_true",
                        help="Sweep over model parameters")
    parser.add_argument("--sweep_seed", action="store_true",
                        help="Sweep over seeds only (without other model params)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds (e.g., '1,42,43,44')")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print experiments without running")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load sweep config
    sweep_config = load_sweep_config(args.config)
    
    # Get WandB project name
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT", "emergent_analogy_sweep")
    
    # Determine which parameters to sweep
    data_sweep = sweep_config.get("data_sweep", {})
    optimizer_sweep = sweep_config.get("optimizer_sweep", {})
    model_sweep = sweep_config.get("model_sweep", {})
    fixed_params = sweep_config.get("fixed", {})
    
    # If neither flag is set, sweep both (data and optimizer)
    if not args.sweep_data and not args.sweep_optimizer and not args.sweep_model and not args.sweep_seed:
        args.sweep_data = True
        args.sweep_optimizer = True
    
    # Generate parameter combinations
    if args.sweep_data:
        if args.mode == "grid":
            data_combos = generate_grid_combinations(data_sweep)
        else:
            data_combos = generate_random_combinations(data_sweep, args.n_samples)
    else:
        # Use default data params
        data_combos = [{
            "num_entities": 10,
            "num_relations": 10000,
            "atomic_ood_ratio": 0.0,
            "compositional_ood_ratio": 0.1,
            "analogical_ood_ratio": 0.1,
            "include_f_inverse": False,
        }]
    
    if args.sweep_optimizer:
        if args.mode == "grid":
            optimizer_combos = generate_grid_combinations(optimizer_sweep)
        else:
            optimizer_combos = generate_random_combinations(optimizer_sweep, args.n_samples)
    else:
        # Use default optimizer params
        optimizer_combos = [{
            "lr": 1e-4,
            "weight_decay": 0.0,
            "batch_size": 64,
        }]

    if args.sweep_model:
        if args.mode == "grid":
            model_combos = generate_grid_combinations(model_sweep)
        else:
            model_combos = generate_random_combinations(model_sweep, args.n_samples)
    elif args.sweep_seed:
        # Seed sweep only (without other model params)
        if args.seeds:
            # Use command-line specified seeds
            seed_list = [int(s.strip()) for s in args.seeds.split(",")]
        else:
            # Use seeds from config (seed_sweep or model_sweep.seed)
            seed_list = sweep_config.get("seed_sweep", {}).get("seed", 
                        model_sweep.get("seed", [42]))
        model_combos = [{"seed": s} for s in seed_list]
    else:
        model_combos = [{}]

    # Generate all experiment combinations
    experiments = []
    for data_params in data_combos:
        for optimizer_params in optimizer_combos:
            for model_params in model_combos:
                experiments.append((data_params, optimizer_params, model_params))
    
    print(f"Total experiments: {len(experiments)}")
    print(f"WandB Project: {wandb_project}")
    print(f"Sweep mode: {args.mode}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE]")
    
    # Run experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, (data_params, optimizer_params, model_params) in enumerate(experiments):
        experiment_id = f"exp_{timestamp}_{i:03d}"
        
        try:
            run_single_experiment(
                data_params=data_params,
                optimizer_params=optimizer_params,
                model_params=model_params,
                fixed_params=fixed_params,
                base_data_dir=args.data_dir,
                base_runs_dir=args.runs_dir,
                wandb_project=wandb_project,
                experiment_id=experiment_id,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"[ERROR] Experiment {experiment_id} failed: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Sweep completed! Total experiments: {len(experiments)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
