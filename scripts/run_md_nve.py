#!/usr/bin/env python3
"""Run a minimal NVE MD rollout using trained MLP forces (velocity-Verlet)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.new_model.models import MLPotential, MLPotentialConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run simple NVE MD with MLP forces")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--npz", type=str, default="")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.5, help="time step in fs-like unit")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, default="outputs/md_nve/rollout.npz")
    return p.parse_args()


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


def kinetic_energy(v: torch.Tensor) -> torch.Tensor:
    # unit masses baseline
    return 0.5 * torch.sum(v * v)


def main() -> None:
    args = parse_args()

    cfg = {}
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    checkpoint = args.checkpoint or cfg.get("checkpoint", "")
    npz_path = args.npz or cfg.get("data_npz", "")
    split = args.split if "--split" in __import__("sys").argv else cfg.get("split", args.split)
    index = args.index if "--index" in __import__("sys").argv else int(cfg.get("index", args.index))

    integ = cfg.get("integrator", {})
    steps = args.steps if "--steps" in __import__("sys").argv else int(integ.get("steps", args.steps))
    dt_cli = args.dt if "--dt" in __import__("sys").argv else float(integ.get("dt", args.dt))
    temp = args.temperature if "--temperature" in __import__("sys").argv else float(integ.get("temperature", args.temperature))
    seed = args.seed if "--seed" in __import__("sys").argv else int(integ.get("seed", args.seed))
    device = args.device if "--device" in __import__("sys").argv else str(cfg.get("device", args.device))
    output = args.output if "--output" in __import__("sys").argv else str(cfg.get("output", args.output))

    if not checkpoint or not npz_path:
        raise ValueError("checkpoint and npz must be provided via CLI or config")

    rng = np.random.default_rng(seed)

    data = np.load(npz_path)
    split_idx = data[f"{split}_idx"]
    raw_i = int(split_idx[index])

    z = torch.tensor(data["z"], dtype=torch.long).unsqueeze(0).to(device)
    R = torch.tensor(data["R"][raw_i], dtype=torch.float32).unsqueeze(0).to(device)

    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model_from_ckpt(ckpt).to(device)

    # Gaussian init velocity, then zero COM drift
    v0 = rng.normal(loc=0.0, scale=np.sqrt(max(temp, 1e-6) / 300.0), size=R.shape)
    v = torch.tensor(v0, dtype=torch.float32, device=device)
    v = v - v.mean(dim=1, keepdim=True)

    traj_R = []
    traj_Ep = []
    traj_Ek = []
    traj_Et = []

    dt = torch.tensor(dt_cli, dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(steps):
            # Need grad for forces; temporarily enable
            with torch.enable_grad():
                R_req = R.detach().clone().requires_grad_(True)
                out = model(z=z, R=R_req, compute_forces=True)
                F = out["forces"].detach()
                Ep = out["energy"].detach()[0]

            v_half = v + 0.5 * dt * F
            R_new = R + dt * v_half

            with torch.enable_grad():
                R_req2 = R_new.detach().clone().requires_grad_(True)
                out2 = model(z=z, R=R_req2, compute_forces=True)
                F_new = out2["forces"].detach()
                Ep_new = out2["energy"].detach()[0]

            v_new = v_half + 0.5 * dt * F_new

            Ek = kinetic_energy(v_new)
            Et = Ep_new + Ek

            traj_R.append(R_new.squeeze(0).cpu().numpy())
            traj_Ep.append(float(Ep_new.cpu()))
            traj_Ek.append(float(Ek.cpu()))
            traj_Et.append(float(Et.cpu()))

            R, v = R_new, v_new

    Et_arr = np.asarray(traj_Et, dtype=np.float64)
    drift = float((Et_arr[-1] - Et_arr[0]) / (abs(Et_arr[0]) + 1e-12))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        R=np.asarray(traj_R, dtype=np.float32),
        E_potential=np.asarray(traj_Ep, dtype=np.float32),
        E_kinetic=np.asarray(traj_Ek, dtype=np.float32),
        E_total=np.asarray(traj_Et, dtype=np.float32),
    )

    summary = {
        "steps": steps,
        "dt": dt_cli,
        "split": split,
        "split_index": index,
        "raw_index": raw_i,
        "energy_total_start": float(Et_arr[0]),
        "energy_total_end": float(Et_arr[-1]),
        "relative_energy_drift": drift,
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
