"""Baseline for the rollout study: a feed-forward one-step predictor.

It sees [o_t, a_t] and predicts o_{t+1} directly, with no latent state and no
recurrence. Applied recursively it is the simplest possible multi-step predictor,
so it isolates "error compounding from an explicit model" from "error compounding
from a learned latent state".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import build_mlp


class OneStepMLP(nn.Module):
    def __init__(self, obs_dim: int = 3, act_dim: int = 1, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = build_mlp([obs_dim + act_dim, hidden, hidden, obs_dim])

        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def set_normalization(self, obs_mean, obs_std) -> None:
        self.obs_mean.copy_(torch.as_tensor(obs_mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(obs_std, dtype=torch.float32))

    def normalize_obs(self, obs_raw: torch.Tensor) -> torch.Tensor:
        return (obs_raw - self.obs_mean) / self.obs_std

    def denormalize_obs(self, obs_norm: torch.Tensor) -> torch.Tensor:
        return obs_norm * self.obs_std + self.obs_mean

    def forward_norm(self, obs_norm: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs_norm, act], dim=-1))

    @torch.no_grad()
    def recursive_rollout(self, obs_raw: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """obs_raw (B, obs_dim), actions (B, H, act_dim) -> predictions (B, H, obs_dim)."""
        preds = []
        obs = obs_raw
        for k in range(actions.shape[1]):
            obs = self.denormalize_obs(self.forward_norm(self.normalize_obs(obs), actions[:, k]))
            preds.append(obs)
        return torch.stack(preds, dim=1)