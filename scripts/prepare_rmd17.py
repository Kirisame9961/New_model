#!/usr/bin/env python3
"""Prepare molecular datasets (rMD17 or OMA-style NPZ subset) for MLP experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


SUPPORTED_DATASETS = {"rmd17", "npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare dataset splits for lightweight MLP training. "
            "Supports rMD17 and OMA-like NPZ subsets."
        )
    )
    parser.add_argument("--config", type=str, default="configs/data_prep_rmd17.yaml")
    parser.add_argument("--dataset-name", type=str, default=None, help="named dataset profile in config")
    parser.add_argument("--molecule", type=str, default=None, help="used only for rMD17 profile")
    parser.add_argument("--max-samples", type=int, default=None, help="max number of samples")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--source-url", type=str, default=None, help="override source URL")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_split(split: SplitConfig) -> None:
    total = split.train_ratio + split.val_ratio + split.test_ratio
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
    for name, ratio in (
        ("train_ratio", split.train_ratio),
        ("val_ratio", split.val_ratio),
        ("test_ratio", split.test_ratio),
    ):
        if ratio <= 0:
            raise ValueError(f"{name} must be > 0, got {ratio}")


def download_file(url: str, target_path: Path) -> None:
    if not url:
        raise ValueError("source_url is empty. Please provide a valid dataset URL in config or CLI.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        with target_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[saved] {target_path}")


def resolve_profile(config: dict[str, Any], dataset_name: str | None) -> dict[str, Any]:
    profiles = config["datasets"]
    if dataset_name is None:
        dataset_name = config["default_dataset"]
    if dataset_name not in profiles:
        raise KeyError(f"dataset profile '{dataset_name}' not found. available: {list(profiles.keys())}")
    profile = dict(profiles[dataset_name])
    profile["name"] = dataset_name
    if profile["type"] not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported profile type: {profile['type']}")
    return profile


def load_arrays(raw_file: Path, profile: dict[str, Any]) -> dict[str, np.ndarray]:
    data = np.load(raw_file)
    key_map = profile["keys"]

    required = ["z", "R", "E", "F"]
    arrays: dict[str, np.ndarray] = {}
    for logical_key in required:
        source_key = key_map[logical_key]
        if source_key not in data:
            raise KeyError(f"key '{source_key}' (for '{logical_key}') not found in {raw_file}")
        arrays[logical_key] = data[source_key]

    optional_keys = ["cell", "pbc", "stress", "virial"]
    for logical_key in optional_keys:
        source_key = key_map.get(logical_key)
        if source_key and source_key in data:
            arrays[logical_key] = data[source_key]

    return arrays


def sample_and_split(n_total: int, max_samples: int, split: SplitConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_use = min(max_samples, n_total)
    rng = np.random.default_rng(split.seed)
    shuffled = np.arange(n_total)
    rng.shuffle(shuffled)
    selected = shuffled[:n_use]

    n_train = int(n_use * split.train_ratio)
    n_val = int(n_use * split.val_ratio)

    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n_use, dtype=np.int64)
    return selected, train_idx, val_idx, test_idx


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))

    profile = resolve_profile(config, args.dataset_name)
    split_cfg = config["split"]
    out_cfg = config["output"]

    split = SplitConfig(
        train_ratio=args.train_ratio if args.train_ratio is not None else float(split_cfg["train_ratio"]),
        val_ratio=args.val_ratio if args.val_ratio is not None else float(split_cfg["val_ratio"]),
        test_ratio=args.test_ratio if args.test_ratio is not None else float(split_cfg["test_ratio"]),
        seed=args.seed if args.seed is not None else int(split_cfg["seed"]),
    )
    validate_split(split)

    if profile["type"] == "rmd17":
        molecule = args.molecule or profile["molecule"]
        raw_filename = profile["raw_filename_template"].format(molecule=molecule)
        source_url = args.source_url or profile["source_url_template"].format(molecule=molecule)
        output_prefix = f"rmd17_{molecule}"
        profile_meta = {"dataset_type": "rmd17", "molecule": molecule}
    else:
        raw_filename = profile["raw_filename"]
        source_url = args.source_url if args.source_url is not None else profile.get("source_url", "")
        output_prefix = profile.get("output_prefix", profile["name"])
        profile_meta = {
            "dataset_type": "npz",
            "family": profile.get("family", "unknown"),
            "note": profile.get("note", ""),
        }

    max_samples = args.max_samples or int(profile["max_samples"])
    raw_dir = Path(out_cfg["raw_dir"])
    processed_dir = Path(out_cfg["processed_dir"])
    raw_file = raw_dir / raw_filename
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.force_download or not raw_file.exists():
        download_file(source_url, raw_file)
    else:
        print(f"[cache] use existing file: {raw_file}")

    arrays = load_arrays(raw_file, profile)
    n_total = arrays["R"].shape[0]
    selected, train_idx, val_idx, test_idx = sample_and_split(n_total=n_total, max_samples=max_samples, split=split)

    n_use = len(selected)
    out_file = processed_dir / f"{output_prefix}_{n_use}.npz"

    payload: dict[str, np.ndarray] = {
        "z": arrays["z"],
        "R": arrays["R"][selected],
        "E": arrays["E"][selected],
        "F": arrays["F"][selected],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "original_indices": selected,
    }
    for key in ("cell", "pbc", "stress", "virial"):
        if key in arrays:
            value = arrays[key]
            payload[key] = value[selected] if value.shape[0] == n_total else value

    np.savez_compressed(out_file, **payload)

    summary = {
        "profile": profile["name"],
        "raw_file": str(raw_file),
        "processed_file": str(out_file),
        "n_total_raw": int(n_total),
        "n_used": int(n_use),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "split_seed": int(split.seed),
        "features": {
            "z_shape": list(arrays["z"].shape),
            "R_shape": list(arrays["R"].shape),
            "E_shape": list(arrays["E"].shape),
            "F_shape": list(arrays["F"].shape),
        },
        "meta": profile_meta,
    }

    summary_file = processed_dir / f"{output_prefix}_{n_use}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[done] dataset prepared")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
