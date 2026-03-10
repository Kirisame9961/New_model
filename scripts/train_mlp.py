#!/usr/bin/env python3
"""Train lightweight MLP potential on processed NPZ dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from torch import nn
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


def build_model(cfg: dict) -> MLPotential:
    model_cfg = MLPotentialConfig(**cfg["model"])
    return MLPotential(model_cfg)


def make_loader(npz_path: str, split: str, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = MolecularNpzDataset(npz_path=npz_path, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], energy_weight: float, force_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
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


def run_epoch(model: MLPotential, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device, energy_weight: float, force_weight: float) -> dict[str, float]:
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
            optimizer.step()

        for k in metrics:
            metrics[k] += step_metrics[k]
        n_batches += 1

    if n_batches == 0:
        return metrics
    return {k: v / n_batches for k, v in metrics.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    train_cfg = cfg["training"]
    npz_path = cfg["data"]["processed_npz"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)))

    energy_weight = float(train_cfg["loss"]["energy_weight"])
    force_weight = float(train_cfg["loss"]["force_weight"])
    epochs = int(train_cfg["epochs"])

    history: list[dict[str, float | int]] = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, energy_weight, force_weight)
        val_metrics = run_epoch(model, val_loader, None, device, energy_weight, force_weight)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_loss_e": train_metrics["loss_e"],
            "train_loss_f": train_metrics["loss_f"],
            "val_loss": val_metrics["loss"],
            "val_loss_e": val_metrics["loss_e"],
            "val_loss_f": val_metrics["loss_f"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg,
            "model_config": asdict(model.config),
        }
        torch.save(ckpt, output_dir / "last.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(ckpt, output_dir / "best.pt")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] training completed. best_val={best_val:.6f}")


if __name__ == "__main__":
    main()
