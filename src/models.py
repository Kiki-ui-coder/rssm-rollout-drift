"""A recurrent state-space world model (RSSM), written from scratch.

This follows the architecture family of PlaNet / DreamerV1-V3:

    deterministic path : h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    stochastic path    : prior  p(z_t | h_t)      -- used when imagining
                         posterior q(z_t | h_t, e_t) with e_t = enc(o_t)
    heads              : decoder p(o_t | h_t, z_t), reward head p(r_t | h_t, z_t)

Latent is a diagonal Gaussian. The original DreamerV1 used that variant; later
versions switched to categorical latents, which changes the KL but not the
rollout-drift question studied here.

Training objective (per timestep, summed over the sequence and averaged):
    L = ||dec(h,z) - o||^2  +  ||rew(h,z) - r||^2  +  beta * KL_balanced(q || p)

KL balancing with free bits (DreamerV3 style):
    dyn = KL(sg(q) || p)          trains the prior to track the posterior
    rep = max(KL(q || sg(p)), f)  trains the posterior towards the prior
    KL_balanced = alpha * dyn + (1 - alpha) * rep

The free-bits floor on `rep` is what keeps the posterior from collapsing onto a
prior that carries no information.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from .utils import build_mlp


def _normal_from_params(params: torch.Tensor, min_std: float = 0.1) -> Normal:
    """Split a (…, 2*d) tensor into mean / softplus-std of a d-dim diagonal Normal."""
    mean, raw_std = params.chunk(2, dim=-1)
    std = F.softplus(raw_std) + min_std
    return Normal(mean, std)


def _detached(dist: Normal) -> Normal:
    return Normal(dist.mean.detach(), dist.stddev.detach())


def kl_balanced(post: Normal, prior: Normal, alpha: float = 0.8, free_bits: float = 1.0) -> torch.Tensor:
    """Per-sample KL in nats, summed over the latent dimensions."""
    dyn = kl_divergence(_detached(post), prior).sum(-1)
    rep = kl_divergence(post, _detached(prior)).sum(-1)
    rep = torch.clamp(rep, min=free_bits)
    return alpha * dyn + (1.0 - alpha) * rep


class Encoder(nn.Module):
    """Observation -> embedding consumed by the posterior."""

    def __init__(self, obs_dim: int, hidden: int, embed_dim: int):
        super().__init__()
        self.net = build_mlp([obs_dim, hidden, embed_dim])

    def forward(self, obs_norm: torch.Tensor) -> torch.Tensor:
        return self.net(obs_norm)


class Decoder(nn.Module):
    """Latent state -> reconstructed (normalized) observation."""

    def __init__(self, deter_dim: int, stoch_dim: int, hidden: int, obs_dim: int):
        super().__init__()
        self.net = build_mlp([deter_dim + stoch_dim, hidden, obs_dim])

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1))


class RSSM(nn.Module):
    """Recurrent state-space model: deterministic GRU state + stochastic latent."""

    def __init__(
        self,
        act_dim: int,
        deter_dim: int = 128,
        stoch_dim: int = 16,
        hidden: int = 128,
        embed_dim: int = 128,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.min_std = min_std

        self.gru = nn.GRUCell(stoch_dim + act_dim, deter_dim)
        self.prior_net = build_mlp([deter_dim, hidden, 2 * stoch_dim])
        self.post_net = build_mlp([deter_dim + embed_dim, hidden, 2 * stoch_dim])

    def initial_state(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch, self.deter_dim, device=device)
        z = torch.zeros(batch, self.stoch_dim, device=device)
        return h, z

    def obs_step(self, h, z, prev_act, embed, sample: bool = True):
        """Teacher-forced step: the posterior sees the real observation."""
        h = self.gru(torch.cat([z, prev_act], dim=-1), h)
        prior = _normal_from_params(self.prior_net(h), self.min_std)
        post = _normal_from_params(self.post_net(torch.cat([h, embed], dim=-1)), self.min_std)
        z = post.rsample() if sample else post.mean
        return h, z, prior, post

    def img_step(self, h, z, prev_act, sample: bool = True):
        """Imagination step: only the prior is available, so error can compound."""
        h = self.gru(torch.cat([z, prev_act], dim=-1), h)
        prior = _normal_from_params(self.prior_net(h), self.min_std)
        z = prior.rsample() if sample else prior.mean
        return h, z, prior


class WorldModel(nn.Module):
    """Encoder + RSSM + decoder + reward head, with input normalization baked in."""

    def __init__(
        self,
        obs_dim: int = 3,
        act_dim: int = 1,
        deter_dim: int = 128,
        stoch_dim: int = 16,
        hidden: int = 128,
        embed_dim: int = 128,
        min_std: float = 0.1,
        kl_alpha: float = 0.8,
        free_bits: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.kl_alpha = kl_alpha
        self.free_bits = free_bits

        self.encoder = Encoder(obs_dim, hidden, embed_dim)
        self.rssm = RSSM(act_dim, deter_dim, stoch_dim, hidden, embed_dim, min_std)
        self.decoder = Decoder(deter_dim, stoch_dim, hidden, obs_dim)
        self.reward_head = build_mlp([deter_dim + stoch_dim, hidden, 1])

        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("rew_mean", torch.zeros(1))
        self.register_buffer("rew_std", torch.ones(1))

    # ---------------------------------------------------------------- scaling
    def set_normalization(self, obs_mean, obs_std, rew_mean, rew_std) -> None:
        self.obs_mean.copy_(torch.as_tensor(obs_mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(obs_std, dtype=torch.float32))
        self.rew_mean.copy_(torch.as_tensor([rew_mean], dtype=torch.float32))
        self.rew_std.copy_(torch.as_tensor([rew_std], dtype=torch.float32))

    def normalize_obs(self, obs_raw: torch.Tensor) -> torch.Tensor:
        return (obs_raw - self.obs_mean) / self.obs_std

    def denormalize_obs(self, obs_norm: torch.Tensor) -> torch.Tensor:
        return obs_norm * self.obs_std + self.obs_mean

    def normalize_rew(self, rew_raw: torch.Tensor) -> torch.Tensor:
        return (rew_raw - self.rew_mean) / self.rew_std

    def denormalize_rew(self, rew_norm: torch.Tensor) -> torch.Tensor:
        return rew_norm * self.rew_std + self.rew_mean

    # ------------------------------------------------------------- primitives
    def initial_state(self, batch: int):
        return self.rssm.initial_state(batch, self.obs_mean.device)

    def encode(self, obs_raw: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.normalize_obs(obs_raw))

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Returns the observation in ORIGINAL units."""
        return self.denormalize_obs(self.decoder(h, z))

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Returns the reward in ORIGINAL units, shape (B,)."""
        return self.denormalize_rew(self.reward_head(torch.cat([h, z], dim=-1))).squeeze(-1)

    # ------------------------------------------------------------------ losses
    def sequence_loss(self, obs: torch.Tensor, act: torch.Tensor, rew: torch.Tensor) -> dict:
        """obs (B,T,obs_dim), act (B,T,act_dim), rew (B,T) -> scalar losses."""
        batch, steps = obs.shape[0], obs.shape[1]
        obs_norm = self.normalize_obs(obs)
        rew_norm = self.normalize_rew(rew)

        # act[t] produced obs[t]; the model at step t must therefore consume act[t-1].
        zeros = torch.zeros(batch, 1, self.act_dim, device=act.device)
        prev_act = torch.cat([zeros, act[:, :-1]], dim=1)

        h, z = self.initial_state(batch)
        rec_loss = obs.new_zeros(())
        rew_loss = obs.new_zeros(())
        kl_loss = obs.new_zeros(())

        for t in range(steps):
            h, z, prior, post = self.rssm.obs_step(h, z, prev_act[:, t], self.encode(obs[:, t]))
            rec_loss = rec_loss + F.mse_loss(self.decoder(h, z), obs_norm[:, t], reduction="mean")
            rew_loss = rew_loss + F.mse_loss(self.reward_head(torch.cat([h, z], -1)).squeeze(-1), rew_norm[:, t])
            kl_loss = kl_loss + kl_balanced(post, prior, self.kl_alpha, self.free_bits).mean()

        return {
            "rec": rec_loss / steps,
            "rew": rew_loss / steps,
            "kl": kl_loss / steps,
            "loss": (rec_loss + rew_loss + kl_loss) / steps,
        }

    @torch.no_grad()
    def open_loop_rollout(self, h, z, actions: torch.Tensor, sample: bool = True) -> dict:
        """Roll the prior forward with no observation feedback.

        actions (B,H,act_dim): the actions actually taken by the environment.
        Returns stacked (H) tensors, aligned so index k-1 corresponds to
        absolute time `t0 + k`, i.e. the k-th step of the rollout.
        """
        obs_hat, rew_hat, hs, zs, priors = [], [], [], [], []
        for k in range(actions.shape[1]):
            h, z, prior = self.rssm.img_step(h, z, actions[:, k], sample=sample)
            obs_hat.append(self.decode(h, z))
            rew_hat.append(self.predict_reward(h, z))
            hs.append(h)
            zs.append(z)
            priors.append(prior)
        return {
            "obs": torch.stack(obs_hat, dim=1),
            "rew": torch.stack(rew_hat, dim=1),
            "h": torch.stack(hs, dim=1),
            "z": torch.stack(zs, dim=1),
            "prior": priors,
        }