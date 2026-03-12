#!/usr/bin/env python3
"""Run energy/force prediction with a trained MLP checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.new_model.models import MLPotential, MLPotentialConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict energy and forces with trained MLP")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--npz", type=str, required=True, help="processed npz file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0, help="index within selected split")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def build_model_from_ckpt(ckpt: dict) -> MLPotential:
    if "model_config" in ckpt:
        cfg = MLPotentialConfig(**ckpt["model_config"])
    elif "config" in ckpt and "model" in ckpt["config"]:
        cfg = MLPotentialConfig(**ckpt["config"]["model"])
    else:
        raise KeyError("Checkpoint missing model config")
    model = MLPotential(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    data = np.load(args.npz)
    split_idx = data[f"{args.split}_idx"]

    if args.index < 0 or args.index >= len(split_idx):
        raise IndexError(f"index={args.index} out of range for split size {len(split_idx)}")

    raw_i = int(split_idx[args.index])
    z = torch.tensor(data["z"], dtype=torch.long).unsqueeze(0)
    R = torch.tensor(data["R"][raw_i], dtype=torch.float32).unsqueeze(0)
    E_true = torch.tensor(data["E"][raw_i], dtype=torch.float32).reshape(-1)
    F_true = torch.tensor(data["F"][raw_i], dtype=torch.float32)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model_from_ckpt(ckpt).to(args.device)

    with torch.enable_grad():
        out = model(z=z.to(args.device), R=R.to(args.device), compute_forces=True)

    E_pred = out["energy"].detach().cpu().reshape(-1)
    F_pred = out["forces"].detach().cpu().squeeze(0)

    result = {
        "split": args.split,
        "split_index": args.index,
        "raw_index": raw_i,
        "energy_true": float(E_true[0]),
        "energy_pred": float(E_pred[0]),
        "energy_abs_error": float(torch.abs(E_pred[0] - E_true[0])),
        "force_mae": float(torch.mean(torch.abs(F_pred - F_true))),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
