"""Shared helpers: seeding, device resolution, small MLP builder, JSON I/O."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch so a run is reproducible from its command line."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str | None = None) -> torch.device:
    """Resolve the compute device.

    Defaults to CPU on purpose: the models here are tiny, and CPU keeps the
    numbers reproducible across machines (MPS kernels are not deterministic).
    """
    name = name or os.environ.get("RSSM_DEVICE", "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def build_mlp(sizes, act=nn.ELU) -> nn.Sequential:
    """Linear stack: activation between layers, linear output layer."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def dump_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)