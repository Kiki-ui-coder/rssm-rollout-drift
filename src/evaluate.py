"""The rollout-drift experiment.

Protocol (identical for every model, so the numbers are comparable):

  1. Collect `--episodes` held-out Pendulum episodes of length `warmup + horizon`
     with deterministic seeds, disjoint from the training data.
  2. Run a teacher-forced pass over the whole episode. This gives, at every real
     timestep t, the posterior state q(z_t | h_t, o_t) -- "what the model would
     believe if it could still see the truth".
  3. Restart from the state at index `warmup - 1` and roll forward `horizon`
     steps using ONLY the prior, feeding the actions the environment really took.
     No observation ever enters.
  4. At each horizon k compare the open-loop prediction against the real future
     (absolute index `warmup - 1 + k`), and compare the open-loop prior against
     the teacher-forced posterior at that same index.

Reported per (model, episode, horizon):
  obs_mse_norm  squared error of the reconstructed observation, in normalized units
  mae_cos / mae_sin / mae_thdot   per-dimension ABSOLUTE error in the ORIGINAL
                                  units; the mean over episodes of these is the MAE
  rew_mse       squared error of the predicted reward, original units
  latent_kl     KL(q_tf || p_open) in nats -- the belief drift
  h_rel_drift   ||h_open - h_tf|| / ||h_tf|| -- deterministic-path drift

Models compared:
  rssm_stochastic      prior sampling during imagination (the standard recipe)
  rssm_mean            prior mean instead of a sample (isolates latent noise)
  rssm_teacher_forced  one-step reference: posterior state, horizon-independent
  mlp_recursive        feed-forward one-step MLP applied recursively (no latent)
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset

from .baselines import OneStepMLP
from .envs import collect_episodes
from .models import WorldModel
from .utils import dump_json, get_device, load_json, set_seed

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure open-loop rollout error accumulation")
    p.add_argument("--ckpt", nargs="+", required=True, help="one or more trained world-model checkpoints")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--warmup", type=int, default=5, help="teacher-forced steps before the open-loop rollout starts")
    p.add_argument("--horizon", type=int, default=15, help="open-loop rollout length")
    p.add_argument("--eval-seed", type=int, default=777)
    p.add_argument("--baseline-buffer", type=str, default="data_cache/train_120000_1234.npz",
                   help="training buffer used to fit the baseline (same one used for the world model)")
    p.add_argument("--baseline-epochs", type=int, default=25)
    p.add_argument("--baseline-lr", type=float, default=1e-3)
    p.add_argument("--baseline-batch", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_world_model(path: str, device) -> tuple[WorldModel, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = WorldModel(
        obs_dim=3,
        act_dim=1,
        deter_dim=cfg["deter_dim"],
        stoch_dim=cfg["stoch_dim"],
        hidden=cfg["hidden"],
        embed_dim=cfg["embed_dim"],
        kl_alpha=cfg["kl_alpha"],
        free_bits=cfg["free_bits"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model, ckpt


def train_baseline(norm: dict, buffer_path: str, args, device, seed: int) -> OneStepMLP:
    if not os.path.exists(buffer_path):
        raise FileNotFoundError(
            f"missing training buffer {buffer_path}; run src.train first so the buffer is cached"
        )
    with np.load(buffer_path) as fh:
        obs, act = fh["obs"], fh["act"]

    set_seed(seed)
    model = OneStepMLP().to(device)
    model.set_normalization(norm["obs_mean"], norm["obs_std"])

    x_obs = torch.from_numpy(obs[:-1])
    x_act = torch.from_numpy(act[:-1])
    y_obs = torch.from_numpy(obs[1:])
    loader = DataLoader(TensorDataset(x_obs, x_act, y_obs), batch_size=args.baseline_batch, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.baseline_lr)
    for _ in range(args.baseline_epochs):
        model.train()
        for b_obs, b_act, b_y in loader:
            b_obs, b_act, b_y = b_obs.to(device), b_act.to(device), b_y.to(device)
            pred = model.forward_norm(model.normalize_obs(b_obs), b_act)
            loss = torch.nn.functional.mse_loss(pred, model.normalize_obs(b_y))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def rssm_protocol(model: WorldModel, episodes, warmup: int, horizon: int, seed_tag: int, device):
    """Run the teacher-forced pass + open-loop rollouts. Returns rows and one example."""
    obs = torch.as_tensor(np.stack([e["obs"] for e in episodes]), device=device)
    act = torch.as_tensor(np.stack([e["act"] for e in episodes]), device=device)
    rew = torch.as_tensor(np.stack([e["rew"] for e in episodes]), device=device)

    batch, T, obs_dim = obs.shape
    i0 = warmup - 1
    stoch = model.rssm.stoch_dim
    deter = model.rssm.deter_dim

    def zeros_seq(dim: int):
        return torch.zeros(batch, T, dim, device=device)

    tf_h = zeros_seq(deter)
    tf_z = zeros_seq(stoch)
    tf_post_m, tf_post_s = zeros_seq(stoch), zeros_seq(stoch)
    tf_obs_hat = torch.zeros(batch, T, obs_dim, device=device)
    tf_rew_hat = torch.zeros(batch, T, device=device)

    h, z = model.initial_state(batch)
    for t in range(T):
        prev_act = act[:, t - 1] if t > 0 else torch.zeros(batch, model.act_dim, device=device)
        h, z, prior, post = model.rssm.obs_step(h, z, prev_act, model.encode(obs[:, t]))
        tf_h[:, t], tf_z[:, t] = h, z
        tf_post_m[:, t], tf_post_s[:, t] = post.mean, post.stddev
        tf_obs_hat[:, t] = model.decode(h, z)
        tf_rew_hat[:, t] = model.predict_reward(h, z)

    actions = act[:, i0 : i0 + horizon]  # a_{i0} ... a_{i0+horizon-1}
    rows = []
    # Full episodes are kept so the example figure can plot the warm-up context too.
    example = {"true_obs": obs.cpu().numpy(), "true_rew": rew.cpu().numpy(),
               "i0": np.asarray(i0), "actions": actions.cpu().numpy()}

    for variant, sample in (("rssm_stochastic", True), ("rssm_mean", False)):
        out = model.open_loop_rollout(tf_h[:, i0], tf_z[:, i0], actions, sample=sample)
        example[f"{variant}_obs"] = out["obs"].cpu().numpy()
        example[f"{variant}_rew"] = out["rew"].cpu().numpy()

        for k in range(1, horizon + 1):
            idx = i0 + k
            err = (out["obs"][:, k - 1] - obs[:, idx]) / model.obs_std  # normalized
            obs_mse_norm = err.pow(2).mean(dim=-1)

            raw_err = (out["obs"][:, k - 1] - obs[:, idx]).abs()  # per episode, per dimension

            rew_mse = (out["rew"][:, k - 1] - rew[:, idx]).pow(2)

            prior = Normal(out["prior"][k - 1].mean, out["prior"][k - 1].stddev)
            post_tf = Normal(tf_post_m[:, idx], tf_post_s[:, idx])
            latent_kl = kl_divergence(post_tf, prior).sum(dim=-1)

            h_tf = tf_h[:, idx]
            h_rel = (out["h"][:, k - 1] - h_tf).norm(dim=-1) / (h_tf.norm(dim=-1) + EPS)

            rows.extend(
                _rows("rssm_stochastic" if sample else "rssm_mean", seed_tag, k, obs_mse_norm,
                      raw_err, rew_mse, latent_kl, h_rel)
            )

    # horizon-independent one-step reference from the (teacher-forced) posterior
    for k in range(1, horizon + 1):
        idx = i0 + k
        err = (tf_obs_hat[:, idx] - obs[:, idx]) / model.obs_std
        obs_mse_norm = err.pow(2).mean(dim=-1)
        raw_err = (tf_obs_hat[:, idx] - obs[:, idx]).abs()
        rew_mse = (tf_rew_hat[:, idx] - rew[:, idx]).pow(2)
        # No latent/deterministic drift against itself: this curve IS the reference.
        nan = torch.full((batch,), float("nan"), device=device)
        rows.extend(_rows("rssm_teacher_forced", seed_tag, k, obs_mse_norm, raw_err,
                          rew_mse, nan, nan))
    return rows, example


@torch.no_grad()
def mlp_protocol(model: OneStepMLP, episodes, warmup: int, horizon: int, seed_tag: int, device):
    obs = torch.as_tensor(np.stack([e["obs"] for e in episodes]), device=device)
    act = torch.as_tensor(np.stack([e["act"] for e in episodes]), device=device)
    rew = torch.as_tensor(np.stack([e["rew"] for e in episodes]), device=device)
    i0 = warmup - 1

    actions = act[:, i0 : i0 + horizon]
    preds = model.recursive_rollout(obs[:, i0], actions)

    rows = []
    for k in range(1, horizon + 1):
        idx = i0 + k
        err = (preds[:, k - 1] - obs[:, idx]) / model.obs_std
        obs_mse_norm = err.pow(2).mean(dim=-1)
        raw_err = (preds[:, k - 1] - obs[:, idx]).abs()
        rew_mse = torch.full((obs.shape[0],), float("nan"), device=device)
        latent_kl = torch.full((obs.shape[0],), float("nan"), device=device)
        h_rel = torch.full((obs.shape[0],), float("nan"), device=device)
        rows.extend(_rows("mlp_recursive", seed_tag, k, obs_mse_norm, raw_err, rew_mse, latent_kl, h_rel))
    return rows


def _rows(model_name, seed, horizon, obs_mse_norm, abs_err, rew_mse, latent_kl, h_rel):
    """Expand per-episode tensors into one CSV row per episode."""
    out = []
    n = obs_mse_norm.shape[0]
    r = abs_err.detach().cpu().numpy()  # (n, obs_dim)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"expected abs errors shaped (n, 3), got {r.shape}")
    o = obs_mse_norm.detach().cpu().numpy()
    rw = rew_mse.detach().cpu().numpy()
    kl = latent_kl.detach().cpu().numpy()
    hr = h_rel.detach().cpu().numpy()
    for i in range(n):
        out.append(
            {
                "model": model_name,
                "seed": seed,
                "episode": i,
                "horizon": horizon,
                "obs_mse_norm": float(o[i]),
                "mae_cos": float(r[i, 0]),
                "mae_sin": float(r[i, 1]),
                "mae_thdot": float(r[i, 2]),
                "rew_mse": float(rw[i]),
                "latent_kl": float(kl[i]),
                "h_rel_drift": float(hr[i]),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    # The `rssm_stochastic` variant samples from the prior, so without a fixed
    # seed every run of this script yields slightly different numbers. Seeding
    # here makes the reported results exactly reproducible.
    set_seed(args.eval_seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    episodes = collect_episodes(
        n_episodes=args.episodes,
        horizon=args.warmup + args.horizon,
        seed=args.eval_seed,
    )
    print(f"[eval] {len(episodes)} episodes, length {episodes[0]['obs'].shape[0]} (warmup={args.warmup}, horizon={args.horizon})")

    all_rows: list[dict] = []
    examples: dict[str, np.ndarray] = {}
    loaded = [(path, *load_world_model(path, device)) for path in args.ckpt]
    norm = {k: np.asarray(v) for k, v in loaded[0][2]["norm"].items()}

    for ckpt_path, model, ckpt in loaded:
        seed = int(ckpt["config"]["seed"])
        print(f"[eval] {ckpt_path} (seed={seed}, best val loss={ckpt.get('best_val_loss'):.4f})")
        rows, example = rssm_protocol(model, episodes, args.warmup, args.horizon, seed, device)
        all_rows.extend(rows)
        for key, value in example.items():
            examples.setdefault(key, value)

    # One baseline per seed keeps the comparison like-for-like.
    for _, _, ckpt in loaded:
        seed = int(ckpt["config"]["seed"])
        baseline = train_baseline(norm, args.baseline_buffer, args, device, seed)
        all_rows.extend(mlp_protocol(baseline, episodes, args.warmup, args.horizon, seed, device))
        print(f"[eval] mlp_recursive trained (seed={seed})")

    raw_path = os.path.join(args.out, "rollout_raw.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    np.savez_compressed(os.path.join(args.out, "example_trajectory.npz"), **examples)

    dump_json(
        os.path.join(args.out, "eval_meta.json"),
        {
            "episodes": args.episodes,
            "warmup": args.warmup,
            "horizon": args.horizon,
            "eval_seed": args.eval_seed,
            "checkpoints": args.ckpt,
            "seeds": sorted({r["seed"] for r in all_rows}),
            "rows": len(all_rows),
        },
    )
    print(f"[eval] wrote {len(all_rows)} rows -> {raw_path}")


if __name__ == "__main__":
    main()