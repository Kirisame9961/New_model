"""Dataset utilities for processed molecular NPZ files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MolecularNpzDataset(Dataset):
    """PyTorch dataset for processed molecular NPZ files.

    Expected keys in processed file:
    - z: [N_atoms]
    - R: [N_samples, N_atoms, 3]
    - E: [N_samples] or [N_samples, 1]
    - F: [N_samples, N_atoms, 3]
    - split index array (train_idx / val_idx / test_idx)
    """

    def __init__(self, npz_path: str | Path, split: str) -> None:
        super().__init__()
        split_key = f"{split}_idx"
        data = np.load(npz_path)

        if split_key not in data:
            raise KeyError(f"Missing split key '{split_key}' in {npz_path}")

        indices = data[split_key].astype(np.int64)
        self.z = data["z"].astype(np.int64)
        self.R = data["R"][indices].astype(np.float32)
        self.E = data["E"][indices].astype(np.float32)
        self.F = data["F"][indices].astype(np.float32)

        # Normalize shape to [B]
        if self.E.ndim == 2 and self.E.shape[1] == 1:
            self.E = self.E[:, 0]

    def __len__(self) -> int:
        return int(self.R.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "z": torch.from_numpy(self.z),
            "R": torch.from_numpy(self.R[index]),
            "E": torch.tensor(self.E[index]),
            "F": torch.from_numpy(self.F[index]),
        }
