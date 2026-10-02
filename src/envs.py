"""Data collection for Pendulum-v1.

Why Pendulum: it is cheap, fully observable, continuous-action, and it has an
analytic-ish regime structure (swing-up then stabilisation), which gives a broad
range of states for the dynamics model to learn. That makes it a reasonable
sandbox for measuring how far a learned latent model can be rolled out before
the prediction drifts, without needing a GPU cluster.

Exploration: actions come from a mixture of a uniform random policy and a simple
energy controller. Pure random actions keep the pole near the bottom, so a
50/50 mixture is used to also cover the near-upright, low-velocity region that
the task actually cares about.
"""

from __future__ import annotations

import numpy as np

import gymnasium as gym

PENDULUM_MAX_TORQUE = 2.0


def _reset(env, seed: int) -> np.ndarray:
    out = env.reset(seed=int(seed))
    obs = out[0] if isinstance(out, tuple) else out
    return np.asarray(obs, dtype=np.float32)


def _step(env, action: np.ndarray):
    out = env.step(action.astype(np.float32))
    obs, reward, terminated, truncated, _ = out
    return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated)


def scripted_action(obs: np.ndarray) -> np.ndarray:
    """PD controller on the wrapped angle: u = -clamp(k*theta + d*theta_dot).

    In Pendulum-v1 the torque enters the angular acceleration with a positive
    coefficient, so a negative feedback term drives theta towards 0 (upright).
    """
    cos_t, sin_t, theta_dot = obs
    theta = float(np.arctan2(sin_t, cos_t))
    torque = -(2.0 * theta + 0.5 * theta_dot)
    torque = float(np.clip(torque, -PENDULUM_MAX_TORQUE, PENDULUM_MAX_TORQUE))
    return np.array([torque], dtype=np.float32)


def collect_dataset(
    n_steps: int,
    seed: int,
    scripted_prob: float = 0.5,
    max_episode_steps: int | None = None,
) -> dict[str, np.ndarray]:
    """Collect a flat transition buffer of (obs, act, rew).

    `act[t]` is the action applied at `obs[t]` and leads to `obs[t + 1]`.
    """
    env = gym.make("Pendulum-v1")
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    rng = np.random.default_rng(seed)

    obs = _reset(env, seed)
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    rew_buf: list[float] = []

    for t in range(n_steps):
        if rng.random() < scripted_prob:
            action = scripted_action(obs)
        else:
            action = rng.uniform(-PENDULUM_MAX_TORQUE, PENDULUM_MAX_TORQUE, size=(1,)).astype(np.float32)

        obs_buf.append(obs)
        act_buf.append(action)
        obs, reward, terminated, truncated = _step(env, action)
        rew_buf.append(reward)
        if terminated or truncated:
            obs = _reset(env, seed + t + 1)

    env.close()
    return {
        "obs": np.stack(obs_buf).astype(np.float32),
        "act": np.stack(act_buf).astype(np.float32),
        "rew": np.asarray(rew_buf, dtype=np.float32),
    }


def collect_episodes(
    n_episodes: int,
    horizon: int,
    seed: int,
    scripted_prob: float = 0.5,
) -> list[dict[str, np.ndarray]]:
    """Collect independent evaluation episodes with deterministic seeds.

    Each episode is a contiguous, unbroken rollout of length `horizon`, so that
    open-loop predictions can be compared against the real future without
    episode resets in the middle.
    """
    episodes = []
    for e in range(n_episodes):
        env = gym.make("Pendulum-v1")
        rng = np.random.default_rng(seed + 1000 * e)
        obs = _reset(env, seed + 1000 * e)

        obs_buf, act_buf, rew_buf = [], [], []
        for _ in range(horizon):
            action = (
                scripted_action(obs)
                if rng.random() < scripted_prob
                else rng.uniform(-PENDULUM_MAX_TORQUE, PENDULUM_MAX_TORQUE, size=(1,)).astype(np.float32)
            )
            obs_buf.append(obs)
            act_buf.append(action)
            obs, reward, _, _ = _step(env, action)
            rew_buf.append(reward)
        env.close()
        episodes.append(
            {
                "obs": np.stack(obs_buf).astype(np.float32),
                "act": np.stack(act_buf).astype(np.float32),
                "rew": np.asarray(rew_buf, dtype=np.float32),
            }
        )
    return episodes