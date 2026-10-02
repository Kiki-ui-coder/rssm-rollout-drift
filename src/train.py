"""Train the RSSM world model on Pendulum-v1.

Example:
    python -m src.train --epochs 60 --seed 0 --out runs/wm_seed0
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SequenceDataset, normalization_stats
from .envs import collect_dataset
from .models import WorldModel
from .utils import count_parameters, dump_json, get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an RSSM world model on Pendulum-v1")
    p.add_argument("--out", type=str, required=True, help="checkpoint directory")
    p.add_argument("--cache-dir", type=str, default="data_cache", help="where the replay buffers are cached")
    p.add_argument("--train-steps", type=int, default=120_000, help="env steps collected for training")
    p.add_argument("--val-steps", type=int, default=20_000, help="env steps collected for validation")
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--stride", type=int, default=2, help="window stride over the training buffer")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--deter-dim", type=int, default=128)
    p.add_argument("--stoch-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--kl-alpha", type=float, default=0.8)
    p.add_argument("--free-bits", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--val-seed", type=int, default=4321)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_or_collect(cache_path: str, n_steps: int, seed: int) -> dict[str, np.ndarray]:
    if os.path.exists(cache_path):
        with np.load(cache_path) as fh:
            return {k: fh[k] for k in fh.files}
    buffer = collect_dataset(n_steps=n_steps, seed=seed)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(cache_path, **buffer)
    return buffer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    train_buf = load_or_collect(os.path.join(args.cache_dir, f"train_{args.train_steps}_{args.data_seed}.npz"),
                                args.train_steps, args.data_seed)
    val_buf = load_or_collect(os.path.join(args.cache_dir, f"val_{args.val_steps}_{args.val_seed}.npz"),
                              args.val_steps, args.val_seed)

    stats = normalization_stats(train_buf["obs"], train_buf["rew"])

    train_ds = SequenceDataset(train_buf["obs"], train_buf["act"], train_buf["rew"], args.seq_len, args.stride)
    val_ds = SequenceDataset(val_buf["obs"], val_buf["act"], val_buf["rew"], args.seq_len, args.stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    model = WorldModel(
        obs_dim=train_buf["obs"].shape[1],
        act_dim=train_buf["act"].shape[1],
        deter_dim=args.deter_dim,
        stoch_dim=args.stoch_dim,
        hidden=args.hidden,
        embed_dim=args.embed_dim,
        kl_alpha=args.kl_alpha,
        free_bits=args.free_bits,
    ).to(device)
    model.set_normalization(stats["obs_mean"], stats["obs_std"], stats["rew_mean"], stats["rew_std"])

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = count_parameters(model)
    print(f"[train] device={device} params={n_params:,} train_windows={len(train_ds):,} val_windows={len(val_ds):,}")

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for obs, act, rew in train_loader:
            losses = model.sequence_loss(obs.to(device), act.to(device), rew.to(device))
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        model.eval()
        val_totals = {"loss": 0.0, "rec": 0.0, "rew": 0.0, "kl": 0.0}
        n_batches = 0
        with torch.no_grad():
            for obs, act, rew in val_loader:
                losses = model.sequence_loss(obs.to(device), act.to(device), rew.to(device))
                for k in val_totals:
                    val_totals[k] += float(losses[k])
                n_batches += 1
        for k in val_totals:
            val_totals[k] /= max(n_batches, 1)

        record = {
            "epoch": epoch,
            "val_loss": val_totals["loss"],
            "val_rec": val_totals["rec"],
            "val_rew": val_totals["rew"],
            "val_kl": val_totals["kl"],
            "secs": round(time.time() - t0, 2),
        }
        history.append(record)

        if val_totals["loss"] < best_val:
            best_val = val_totals["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(args),
                    "norm": {k: np.asarray(v) for k, v in stats.items()},
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                },
                os.path.join(args.out, "checkpoint.pt"),
            )

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[train] epoch {epoch:3d}/{args.epochs} "
                f"val_loss={val_totals['loss']:.4f} rec={val_totals['rec']:.4f} "
                f"rew={val_totals['rew']:.4f} kl={val_totals['kl']:.4f} ({record['secs']}s)"
            )

    dump_json(os.path.join(args.out, "history.json"), {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val})
    print(f"[train] done. best val_loss={best_val:.4f} at epoch {best_epoch}. ckpt -> {args.out}/checkpoint.pt")


if __name__ == "__main__":
    main()