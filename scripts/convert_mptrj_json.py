#!/usr/bin/env python3
"""Convert MPtrj JSON (figshare export) into standardized NPZ for this project.

Output NPZ keys:
- z: [N_atoms]
- R: [N_samples, N_atoms, 3]
- E: [N_samples]
- F: [N_samples, N_atoms, 3]
- optional: cell [N_samples, 3, 3], pbc [N_samples, 3]

Notes:
- This baseline pipeline assumes fixed atom count/order across selected frames.
- For multi-material mixed JSON, use --group-key (e.g. material_id) to select one group.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MPtrj JSON to standardized NPZ")
    parser.add_argument("--input", type=str, required=True, help="path to MPtrj_*.json")
    parser.add_argument("--output", type=str, required=True, help="output npz path")
    parser.add_argument("--group-key", type=str, default="material_id", help="group key to keep one trajectory/material")
    parser.add_argument("--group-value", type=str, default=None, help="specific group value to keep")
    parser.add_argument("--max-samples", type=int, default=None, help="optional max samples after filtering")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pick(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def to_records(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in ("data", "records", "entries", "trajectories"):
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # fallback: dict-of-records
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
    raise ValueError("Unsupported JSON structure. Expect list[dict] or dict containing list field.")


def infer_group_value(records: list[dict[str, Any]], group_key: str) -> str | None:
    values = [str(r[group_key]) for r in records if group_key in r]
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    obj = json.loads(input_path.read_text(encoding="utf-8"))
    records = to_records(obj)
    if not records:
        raise ValueError("No valid records in input JSON")

    # group selection (important for fixed-atom baseline pipeline)
    group_value = args.group_value or infer_group_value(records, args.group_key)
    if group_value is not None:
        records = [r for r in records if str(r.get(args.group_key, "")) == str(group_value)]

    if not records:
        raise ValueError("No records left after group filtering")

    frames: list[np.ndarray] = []
    energies: list[float] = []
    forces: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    pbcs: list[np.ndarray] = []
    z_ref: np.ndarray | None = None

    for rec in records:
        z = pick(rec, ["atomic_numbers", "z", "numbers", "species_numbers"])
        R = pick(rec, ["positions", "R", "coords", "cart_coords"])
        E = pick(rec, ["energy", "E", "total_energy", "y"])
        F = pick(rec, ["forces", "F", "force", "gradients"])

        if z is None or R is None or E is None or F is None:
            continue

        z_arr = np.asarray(z, dtype=np.int64)
        R_arr = np.asarray(R, dtype=np.float32)
        F_arr = np.asarray(F, dtype=np.float32)

        if z_ref is None:
            z_ref = z_arr
        # enforce fixed atom ordering/size for this baseline
        if z_ref.shape != z_arr.shape or not np.array_equal(z_ref, z_arr):
            continue

        if R_arr.shape != F_arr.shape or R_arr.ndim != 2 or R_arr.shape[-1] != 3:
            continue

        frames.append(R_arr)
        forces.append(F_arr)
        energies.append(float(np.asarray(E).reshape(-1)[0]))

        cell = pick(rec, ["cell", "lattice", "lattice_vectors"], default=None)
        if cell is not None:
            cell_arr = np.asarray(cell, dtype=np.float32)
            if cell_arr.shape == (3, 3):
                cells.append(cell_arr)

        pbc = pick(rec, ["pbc"], default=None)
        if pbc is not None:
            pbc_arr = np.asarray(pbc, dtype=np.bool_)
            if pbc_arr.shape == (3,):
                pbcs.append(pbc_arr)

    if z_ref is None or not frames:
        raise ValueError(
            "Could not extract frames with required fields z/R/E/F and fixed atom topology. "
            "Try a different --group-key/--group-value."
        )

    n_total = len(frames)
    idx = np.arange(n_total)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)

    if args.max_samples is not None:
        idx = idx[: min(args.max_samples, n_total)]

    R = np.asarray(frames, dtype=np.float32)[idx]
    E = np.asarray(energies, dtype=np.float32)[idx]
    F = np.asarray(forces, dtype=np.float32)[idx]

    payload: dict[str, np.ndarray] = {"z": z_ref, "R": R, "E": E, "F": F}

    if len(cells) == n_total:
        payload["cell"] = np.asarray(cells, dtype=np.float32)[idx]
    if len(pbcs) == n_total:
        payload["pbc"] = np.asarray(pbcs, dtype=np.bool_)[idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "group_key": args.group_key,
        "group_value": group_value,
        "n_records_input": int(len(records)),
        "n_samples_output": int(R.shape[0]),
        "z_shape": list(z_ref.shape),
        "R_shape": list(R.shape),
        "E_shape": list(E.shape),
        "F_shape": list(F.shape),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
