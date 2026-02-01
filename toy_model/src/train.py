"""
Training script for Emergent Analogy experiment.
"""

import argparse
import json
import math
import os
import random
import time
from functools import partial

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data import CompDataset, collate_pad
from model import GPT2LikeEncoder


class WarmupThenConstant(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with warmup followed by constant rate."""
    
    def __init__(self, opt, warmup_steps=2000, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(opt, last_epoch)
    
    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = step / self.warmup_steps if step <= self.warmup_steps else 1.0
        return [base * scale for base in self.base_lrs]


def step_loss(model, batch, device, use_amp=True):
    """Compute loss and accuracy for a batch."""
    input_ids = batch["input_ids"].to(device)
    target_ids = batch["target_ids"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    pad_mask = batch.get("pad_mask", None)
    if pad_mask is not None:
        pad_mask = pad_mask.to(device)

    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = model(input_ids, pad_mask=pad_mask)
        B, L, V = logits.shape
        logits_f = logits.view(B * L, V)
        targets_f = target_ids.view(B * L)
        mask_f = loss_mask.view(B * L)
        sel_logits = logits_f[mask_f]
        sel_targets = targets_f[mask_f]
        loss = F.cross_entropy(sel_logits, sel_targets)

    with torch.no_grad():
        pred = sel_logits.argmax(dim=-1)
        acc = (pred == sel_targets).float().mean().item()
        # Calculate probability of correct token
        probs = F.softmax(sel_logits, dim=-1)
        prob = probs[torch.arange(len(sel_targets), device=sel_logits.device), sel_targets].mean().item()
    
    return loss, acc, prob


@torch.no_grad()
def evaluate_loader(model, loader, device, use_amp=True):
    """Evaluate model on a data loader."""
    if loader is None:
        return None
    model.eval()
    sum_loss, sum_acc, n = 0.0, 0.0, 0
    for batch in loader:
        loss, acc, _ = step_loss(model, batch, device, use_amp)
        bs = batch["input_ids"].size(0)
        sum_loss += loss.item() * bs
        sum_acc += acc * bs
        n += bs
    mean_ce = sum_loss / max(1, n)
    return {
        "CE": mean_ce,
        "PPL": math.exp(min(20.0, mean_ce)),
        "ACC": sum_acc / max(1, n),
        "N": n
    }


@torch.no_grad()
def evaluate_split_by_type(model, loader, device, use_amp=True):
    """Evaluate model and split results by data type."""
    model.eval()
    sums = {}
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        types = batch["type"]

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, pad_mask=pad_mask)
        
        B, L, V = logits.shape
        logits_f = logits.view(B * L, V)
        targets_f = target_ids.view(B * L)
        mask_f = loss_mask.view(B * L)
        sel_logits = logits_f[mask_f]
        sel_targets = targets_f[mask_f]
        
        per_ce = F.cross_entropy(sel_logits, sel_targets, reduction='none')
        per_acc = (sel_logits.argmax(dim=-1) == sel_targets).float()
        
        # Calculate probability of correct token
        probs = F.softmax(sel_logits, dim=-1)
        per_prob = probs[torch.arange(len(sel_targets)), sel_targets]
        
        for t, ce, acc, prob in zip(types, per_ce.tolist(), per_acc.tolist(), per_prob.tolist()):
            d = sums.setdefault(t, {"sum_ce": 0, "sum_acc": 0, "sum_prob": 0, "n": 0})
            d["sum_ce"] += ce
            d["sum_acc"] += acc
            d["sum_prob"] += prob
            d["n"] += 1
    
    metrics = {}
    for t, d in sums.items():
        mce = d["sum_ce"] / max(1, d["n"])
        metrics[t] = {
            "CE": mce,
            "PPL": math.exp(min(20.0, mce)),
            "ACC": d["sum_acc"] / max(1, d["n"]),
            "PROB": d["sum_prob"] / max(1, d["n"]),
            "N": d["n"]
        }
    return metrics


def flatten_metrics(prefix, mdict):
    """Flatten metrics dictionary for logging."""
    flat = {}
    for t, m in mdict.items():
        for k, v in m.items():
            flat[f"{prefix}_{k}/{t}"] = v
    return flat


def train(config):
    """Main training function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    
    os.makedirs(config["save_dir"], exist_ok=True)
    
    # Data loading
    data_dir = config["data_dir"]
    vocab_path = os.path.join(data_dir, "vocab.json")
    train_path = os.path.join(data_dir, "train.json")
    test_path = os.path.join(data_dir, "test.json")
    
    train_ds = CompDataset(train_path, vocab_path, max_len=config["max_len"], expect_type=False)
    test_ds = CompDataset(test_path, vocab_path, max_len=config["max_len"], expect_type=True)
    
    collate_fn = partial(collate_pad, pad_id=0)
    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Model setup
    model = GPT2LikeEncoder(
        len(train_ds.vocab),
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        dropout=config["dropout"],
        max_len=config["max_len"]
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = WarmupThenConstant(optimizer, warmup_steps=config["warmup_steps"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
    
    # Optional W&B logging
    wandb = None
    if config.get("use_wandb", False):
        try:
            import wandb as _wandb
            wandb = _wandb
            mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
            
            # Get project/run name from env vars (priority: env > config > default)
            wandb_project = os.environ.get("WANDB_PROJECT") or config.get("project", "emergent_analogy")
            wandb_run_name = os.environ.get("WANDB_RUN_NAME") or config.get("run_name") or f"run_{int(time.time())}"
            
            print(f"[W&B] project: {wandb_project}, run: {wandb_run_name}")
            
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                mode=mode,
                config=config,
            )
            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="global_step")
            wandb.define_metric("lr", step_metric="global_step")
        except Exception as e:
            print(f"[W&B] disabled: {e}")
            wandb = None
    
    # Training loop
    best_val = float("inf")
    global_step = 0
    log_every = config.get("log_every", 50)
    
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running, n_ex = 0, 0
        
        for i, batch in enumerate(train_dl, 1):
            loss, acc, prob = step_loss(model, batch, device, config["use_amp"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            bs = batch["input_ids"].size(0)
            running += loss.item() * bs
            n_ex += bs
            global_step += 1
            
            if wandb and (i % log_every == 0 or i == 1):
                wandb.log({
                    "global_step": global_step,
                    "train/CE_last": loss.item(),
                    "train/ACC_last": acc,
                    "train/PROB_last": prob,
                    "lr": optimizer.param_groups[0]["lr"]
                }, step=global_step)
        
        train_ce = running / max(1, n_ex)
        
        if epoch % config.get("eval_every", 10) == 0:
            print(f"epoch {epoch} | train CE: {train_ce:.4f}")
        
        # Evaluation
        metrics = evaluate_split_by_type(model, test_dl, device, config["use_amp"])
        ce_macro = sum(m["CE"] for m in metrics.values()) / max(1, len(metrics))
        acc_macro = sum(m["ACC"] for m in metrics.values()) / max(1, len(metrics))
        
        if epoch % config.get("eval_every", 10) == 0:
            print(f"\n== epoch {epoch} validation ==")
            for k, v in sorted(metrics.items()):
                print(f"{k:>25}: CE={v['CE']:.4f} | PPL={v['PPL']:.3f} | ACC={v['ACC']:.3f} | PROB={v['PROB']:.4f} | N={v['N']}")
            print(f"{'macro(CE)':>25}: CE={ce_macro:.4f}")
            print(f"{'macro(ACC)':>25}: ACC={acc_macro:.3f}\n")
        
        if wandb:
            log_payload = {
                "global_step": global_step,
                "epoch": epoch,
                "train/CE_epoch": train_ce,
                "val/macro/CE": ce_macro,
                "val/macro/ACC": acc_macro
            }
            log_payload.update(flatten_metrics("val", metrics))
            wandb.log(log_payload, step=global_step)
        
        # Save checkpoint (skip if save_every <= 0)
        save_every = config.get("save_every", 0)
        if save_every > 0:
            ckpt = {
                "model": model.state_dict(),
                "config": {
                    "d_model": config["d_model"],
                    "n_layer": config["n_layer"],
                    "n_head": config["n_head"],
                    "dropout": config["dropout"],
                    "max_len": config["max_len"],
                },
                "vocab": train_ds.vocab,
                "epoch": epoch,
            }
            
            if epoch % save_every == 0:
                torch.save(ckpt, os.path.join(config["save_dir"], f"epoch{epoch:03d}.pt"))
            
            if ce_macro < best_val:
                best_val = ce_macro
                torch.save(ckpt, os.path.join(config["save_dir"], "best.pt"))
    
    if wandb:
        wandb.finish()
    
    print("Training completed!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Emergent Analogy model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override save directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name")
    parser.add_argument("--wandb_run", type=str, default=None,
                        help="WandB run name")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.save_dir:
        config["save_dir"] = args.save_dir
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["lr"] = args.lr
    if args.no_wandb:
        config["use_wandb"] = False
    if args.wandb_project:
        config["project"] = args.wandb_project
    if args.wandb_run:
        config["run_name"] = args.wandb_run
    
    train(config)


if __name__ == "__main__":
    main()
