#!/usr/bin/env python3
"""Evaluate trained MLP potential on processed NPZ test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.new_model.data import MolecularNpzDataset
from src.new_model.models import MLPotential, MLPotentialConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lightweight MLP potential")
    parser.add_argument("--config", type=str, default="configs/eval_mlp.yaml")
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).detach().cpu())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).detach().cpu())


def build_model(ckpt: dict, model_override: dict | None) -> MLPotential:
    if model_override is not None:
        cfg = MLPotentialConfig(**model_override)
    elif "model_config" in ckpt:
        cfg = MLPotentialConfig(**ckpt["model_config"])
    elif "config" in ckpt and "model" in ckpt["config"]:
        cfg = MLPotentialConfig(**ckpt["config"]["model"])
    else:
        raise KeyError("Cannot infer model config from checkpoint. Please provide config.model_override")

    model = MLPotential(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    npz_path = cfg["data"]["processed_npz"]
    ckpt_path = cfg["checkpoint"]["path"]
    split = cfg["data"].get("split", "test")
    batch_size = int(cfg["eval"].get("batch_size", 8))
    num_workers = int(cfg["eval"].get("num_workers", 0))
    device = torch.device(cfg["eval"].get("device", "cpu"))

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ckpt=ckpt, model_override=cfg.get("model_override"))
    model.to(device)

    dataset = MolecularNpzDataset(npz_path=npz_path, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    e_pred_all: list[torch.Tensor] = []
    e_true_all: list[torch.Tensor] = []
    f_pred_all: list[torch.Tensor] = []
    f_true_all: list[torch.Tensor] = []

    for batch in loader:
        z = batch["z"].to(device)
        R = batch["R"].to(device)
        E = batch["E"].to(device)
        F = batch["F"].to(device)

        outputs = model(z=z, R=R, compute_forces=True)

        e_pred_all.append(outputs["energy"].detach().cpu())
        e_true_all.append(E.detach().cpu())
        f_pred_all.append(outputs["forces"].detach().cpu())
        f_true_all.append(F.detach().cpu())

    e_pred = torch.cat(e_pred_all, dim=0)
    e_true = torch.cat(e_true_all, dim=0)
    f_pred = torch.cat(f_pred_all, dim=0)
    f_true = torch.cat(f_true_all, dim=0)

    results = {
        "split": split,
        "num_samples": int(e_true.shape[0]),
        "energy_mae": mae(e_pred, e_true),
        "energy_rmse": rmse(e_pred, e_true),
        "force_mae": mae(f_pred, f_true),
        "force_rmse": rmse(f_pred, f_true),
    }

    out_dir = Path(cfg["eval"].get("output_dir", "outputs/eval_mlp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
