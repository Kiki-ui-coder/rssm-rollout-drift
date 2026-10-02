"""Sequence dataset built from a flat transition buffer."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Fixed-length windows of (obs, act, rew) for recurrent-model training.

    Actions are NOT shifted here. Inside the model, `act[t]` is the action that
    produced `obs[t]`, so the training loop shifts them when consuming.
    """

    def __init__(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, seq_len: int, stride: int = 1):
        assert obs.shape[0] == act.shape[0] == rew.shape[0], "buffer arrays must be aligned"
        self.obs = obs
        self.act = act
        self.rew = rew
        self.seq_len = seq_len
        self.starts = np.arange(0, len(obs) - seq_len, stride, dtype=np.int64)
        if len(self.starts) == 0:
            raise ValueError(f"buffer of {len(obs)} steps is shorter than seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        e = s + self.seq_len
        return (
            torch.from_numpy(self.obs[s:e]),
            torch.from_numpy(self.act[s:e]),
            torch.from_numpy(self.rew[s:e]),
        )


def normalization_stats(obs: np.ndarray, rew: np.ndarray) -> dict[str, np.ndarray]:
    """Mean/std used to put observations and rewards on a comparable scale.

    std is floored at 1e-6 so that degenerate buffers cannot divide by zero.
    """
    return {
        "obs_mean": obs.mean(axis=0).astype(np.float32),
        "obs_std": np.maximum(obs.std(axis=0), 1e-6).astype(np.float32),
        "rew_mean": np.float32(rew.mean()),
        "rew_std": np.float32(max(float(rew.std()), 1e-6)),
    }