"""Turn results/rollout_raw.csv into the figures used in the report.

Aggregation: rows are individual (model, seed, episode, horizon) measurements.
Curves are the mean over episodes and seeds; the shaded band is +/- 1 std across
those same samples, so it captures both episode-to-episode and seed-to-seed spread.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_STYLE = {
    "rssm_stochastic": ("#1f77b4", "RSSM, prior sampling"),
    "rssm_mean": ("#ff7f0e", "RSSM, prior mean"),
    "rssm_teacher_forced": ("#2ca02c", "RSSM, teacher-forced (1-step ref.)"),
    "mlp_recursive": ("#d62728", "MLP one-step, applied recursively"),
}
PLOT_ORDER = ["rssm_stochastic", "rssm_mean", "mlp_recursive", "rssm_teacher_forced"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot rollout-drift curves")
    p.add_argument("--raw", type=str, default="results/rollout_raw.csv")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--example", type=str, default="results/example_trajectory.npz")
    return p.parse_args()


def load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for key in ("horizon", "seed", "episode"):
            r[key] = int(r[key])
        for key in ("obs_mse_norm", "mae_cos", "mae_sin", "mae_thdot", "rew_mse", "latent_kl", "h_rel_drift"):
            r[key] = float(r[key])
    return rows


def aggregate(rows, metric: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    buckets: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        value = r[metric]
        if np.isnan(value):
            continue
        buckets[r["model"]][r["horizon"]].append(value)

    series = {}
    for model, per_h in buckets.items():
        hs = sorted(per_h)
        mean = np.array([np.mean(per_h[h]) for h in hs])
        std = np.array([np.std(per_h[h]) for h in hs])
        series[model] = (np.array(hs), mean, std)
    return series


def curve_panel(ax, series, title, ylabel, logy=False, zero_floor=False):
    for model in PLOT_ORDER:
        if model not in series:
            continue
        color, label = MODEL_STYLE[model]
        hs, mean, std = series[model]
        ax.plot(hs, mean, marker="o", ms=4, color=color, label=label)
        ax.fill_between(hs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("open-loop horizon (steps)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if logy:
        ax.set_yscale("log")
    if zero_floor:
        ax.set_ylim(bottom=0.0)


def write_summary(path: str, rows, metrics) -> None:
    aggregated = {m: aggregate(rows, m) for m in metrics}
    models = sorted({r["model"] for r in rows})
    horizons = sorted({r["horizon"] for r in rows})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["model", "horizon"] + [f"{m}_mean" for m in metrics]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            for h in horizons:
                row = {"model": model, "horizon": h}
                for m in metrics:
                    hs, mean, _ = aggregated[m].get(model, (np.array([]), np.array([]), np.array([])))
                    if len(hs) and h in hs:
                        row[f"{m}_mean"] = float(mean[list(hs).index(h)])
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.raw)
    os.makedirs(args.out, exist_ok=True)

    metrics = ["obs_mse_norm", "mae_thdot", "mae_cos", "mae_sin", "latent_kl", "h_rel_drift", "rew_mse"]
    write_summary(os.path.join(args.out, "rollout_summary.csv"), rows, metrics)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    curve_panel(axes[0, 0], aggregate(rows, "obs_mse_norm"),
                "Observation prediction error", "normalized MSE", zero_floor=True)
    curve_panel(axes[0, 1], aggregate(rows, "mae_thdot"),
                "Angular-velocity error", r"MAE ($\dot\theta$, rad/s)", zero_floor=True)
    curve_panel(axes[1, 0], aggregate(rows, "latent_kl"),
                "Belief drift: KL(posterior$_{tf}$ || prior$_{open}$)", "KL (nats)", logy=True)
    curve_panel(axes[1, 1], aggregate(rows, "h_rel_drift"),
                "Deterministic-state drift", r"$||h_{open}-h_{tf}||\,/\,||h_{tf}||$", logy=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Open-loop rollout error accumulation, Pendulum-v1 (mean $\\pm$ 1 std)", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(os.path.join(args.out, "rollout_drift.png"), dpi=160)
    print(f"[plot] wrote {os.path.join(args.out, 'rollout_drift.png')}")

    if os.path.exists(args.example):
        data = np.load(args.example)
        i0 = int(data["i0"])
        steps = np.arange(data["true_obs"].shape[1])
        horizon = data["rssm_stochastic_obs"].shape[1]
        # Rollout step k-1 lands on absolute index i0 + k.
        pred_steps = np.arange(i0 + 1, i0 + 1 + horizon)
        fig2, axes2 = plt.subplots(1, 3, figsize=(13, 3.6))
        dims = [("cos $\\theta$", 0), ("sin $\\theta$", 1), ("$\\dot\\theta$", 2)]
        for ax, (name, d) in zip(axes2, dims):
            ax.plot(steps, data["true_obs"][0, :, d], "k-", lw=2, label="ground truth")
            for model in ("rssm_stochastic", "mlp_recursive"):
                key = f"{model}_obs"
                if key in data:
                    color, label = MODEL_STYLE[model]
                    ax.plot(pred_steps, data[key][0, :, d], "--", color=color, label=label)
            ax.axvline(i0, color="gray", ls=":", lw=1)
            ax.set_title(name)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        axes2[0].set_ylabel("observation")
        axes2[2].legend(fontsize=8, loc="best")
        fig2.suptitle("Example episode: open-loop prediction from the warm-up state (dotted line)", fontsize=12)
        fig2.tight_layout(rect=(0, 0, 1, 0.93))
        fig2.savefig(os.path.join(args.out, "example_trajectory.png"), dpi=160)
        print(f"[plot] wrote {os.path.join(args.out, 'example_trajectory.png')}")


if __name__ == "__main__":
    main()