#!/usr/bin/env python3
"""Train lightweight MLP potential on processed NPZ dataset."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.new_model.data import MolecularNpzDataset
from src.new_model.models import MLPotential, MLPotentialConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight MLP potential")
    parser.add_argument("--config", type=str, default="configs/train_mlp.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> MLPotential:
    model_cfg = MLPotentialConfig(**cfg["model"])
    return MLPotential(model_cfg)


def make_loader(npz_path: str, split: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = MolecularNpzDataset(npz_path=npz_path, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    energy_weight: float,
    force_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    e_pred = outputs["energy"]
    f_pred = outputs["forces"]

    e_true = batch["E"]
    f_true = batch["F"]

    loss_e = nn.functional.l1_loss(e_pred, e_true)
    loss_f = nn.functional.l1_loss(f_pred, f_true)
    total = energy_weight * loss_e + force_weight * loss_f
    return total, {
        "loss": float(total.detach().cpu()),
        "loss_e": float(loss_e.detach().cpu()),
        "loss_f": float(loss_f.detach().cpu()),
    }


def run_epoch(
    model: MLPotential,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    energy_weight: float,
    force_weight: float,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    metrics = {"loss": 0.0, "loss_e": 0.0, "loss_f": 0.0}
    n_batches = 0

    for batch in loader:
        batch = to_device(batch, device)
        outputs = model(z=batch["z"], R=batch["R"], compute_forces=True)
        total_loss, step_metrics = compute_loss(outputs, batch, energy_weight, force_weight)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        for k in metrics:
            metrics[k] += step_metrics[k]
        n_batches += 1

    if n_batches == 0:
        return metrics
    return {k: v / n_batches for k, v in metrics.items()}


def save_checkpoint(
    path: Path,
    model: MLPotential,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val: float,
    cfg: dict,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "config": cfg,
        "model_config": asdict(model.config),
    }
    torch.save(ckpt, path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    train_cfg = cfg["training"]
    npz_path = cfg["data"]["processed_npz"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(train_cfg.get("seed", 42)))

    device = torch.device(train_cfg.get("device", "cpu"))
    model = build_model(cfg).to(device)

    train_loader = make_loader(
        npz_path=npz_path,
        split="train",
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 0)),
        shuffle=True,
    )
    val_loader = make_loader(
        npz_path=npz_path,
        split="val",
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 0)),
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg.get("lr_decay_factor", 0.5)),
        patience=int(train_cfg.get("lr_decay_patience", 5)),
        min_lr=float(train_cfg.get("min_lr", 1e-6)),
    )

    energy_weight = float(train_cfg["loss"]["energy_weight"])
    force_weight = float(train_cfg["loss"]["force_weight"])
    epochs = int(train_cfg["epochs"])
    grad_clip_norm = train_cfg.get("grad_clip_norm", None)
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None

    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_epoch = 0
    patience = int(train_cfg.get("early_stopping_patience", 20))
    resume_path = train_cfg.get("resume_checkpoint", "")
    start_epoch = 1

    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"[resume] from {resume_path}, start_epoch={start_epoch}, best_val={best_val:.6f}")

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            energy_weight,
            force_weight,
            grad_clip_norm,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            energy_weight,
            force_weight,
            grad_clip_norm=None,
        )

        scheduler.step(val_metrics["loss"])
        lr_now = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_metrics["loss"],
            "train_loss_e": train_metrics["loss_e"],
            "train_loss_f": train_metrics["loss_f"],
            "val_loss": val_metrics["loss"],
            "val_loss_e": val_metrics["loss_e"],
            "val_loss_f": val_metrics["loss_f"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, best_val, cfg)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, best_val, cfg)

        if patience > 0 and (epoch - best_epoch) >= patience:
            print(f"[early-stop] no val improvement for {patience} epochs. stop at epoch={epoch}")
            break

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] training completed. best_val={best_val:.6f}, best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
