"""Lightweight MLP potential baseline (phase 1 skeleton).

This module provides a small, readable baseline to predict molecular energy and forces:
- Energy is built from per-atom features and aggregated by summation.
- Forces are obtained as negative gradient of energy wrt coordinates.

Current scope intentionally stays minimal for fast iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MLPotentialConfig:
    """Hyperparameters for the baseline MLP potential."""

    num_elements: int = 100
    emb_dim: int = 32
    radial_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 3
    cutoff: float = 5.0


class RadialBasisLayer(nn.Module):
    """Simple Gaussian radial basis expansion for pair distances."""

    def __init__(self, radial_dim: int, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, radial_dim)
        gamma = torch.full((radial_dim,), 10.0 / max(cutoff, 1e-6))
        self.register_buffer("centers", centers)
        self.register_buffer("gamma", gamma)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # distances: [B, N, N]
        diff = distances.unsqueeze(-1) - self.centers
        basis = torch.exp(-self.gamma * diff * diff)
        cutoff_mask = (distances <= self.cutoff).unsqueeze(-1)
        return basis * cutoff_mask


class AtomMLP(nn.Module):
    """Per-atom MLP that maps atom features to atomic energy contributions."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, atom_features: torch.Tensor) -> torch.Tensor:
        # atom_features: [B, N, D]
        return self.net(atom_features).squeeze(-1)  # [B, N]


class MLPotential(nn.Module):
    """A lightweight, geometry-aware baseline MLP potential."""

    def __init__(self, config: MLPotentialConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.num_elements, config.emb_dim)
        self.rbf = RadialBasisLayer(radial_dim=config.radial_dim, cutoff=config.cutoff)
        self.rbf_proj = nn.Linear(config.radial_dim, config.emb_dim)
        self.atom_mlp = AtomMLP(
            input_dim=config.emb_dim * 2,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )

    def _build_atom_features(self, z: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Construct per-atom features from element embeddings + radial neighborhood context.

        Args:
            z: atomic numbers, shape [B, N]
            R: coordinates, shape [B, N, 3]
        Returns:
            atom_features: shape [B, N, 2*emb_dim]
        """
        emb = self.embedding(z)  # [B, N, emb_dim]

        diff = R[:, :, None, :] - R[:, None, :, :]  # [B, N, N, 3]
        dist = torch.linalg.norm(diff, dim=-1)  # [B, N, N]
        rbf = self.rbf(dist)  # [B, N, N, radial_dim]
        neigh_context = self.rbf_proj(rbf).sum(dim=2)  # [B, N, emb_dim]

        return torch.cat([emb, neigh_context], dim=-1)

    def energy(self, z: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Predict total energy for each structure.

        Args:
            z: [B, N]
            R: [B, N, 3]
        Returns:
            E: [B]
        """
        atom_features = self._build_atom_features(z=z, R=R)
        atom_energy = self.atom_mlp(atom_features)  # [B, N]
        return atom_energy.sum(dim=-1)

    def forward(self, z: torch.Tensor, R: torch.Tensor, compute_forces: bool = True) -> dict[str, torch.Tensor]:
        """Run forward pass and optionally compute forces.

        Forces are defined as: F = -dE/dR
        """
        if compute_forces:
            R = R.requires_grad_(True)

        E = self.energy(z=z, R=R)
        outputs: dict[str, torch.Tensor] = {"energy": E}

        if compute_forces:
            grad_outputs = torch.ones_like(E)
            dE_dR = torch.autograd.grad(
                outputs=E,
                inputs=R,
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            outputs["forces"] = -dE_dR

        return outputs
