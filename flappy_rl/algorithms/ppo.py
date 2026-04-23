"""PPO — Proximal Policy Optimization (actor-critic, GAE, clipped surrogate)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .base import BaseAlgorithm, build_mlp, obs_to_tensor, DEVICE


class PPO(BaseAlgorithm):

    def __init__(
        self,
        env,
        learning_rate:   float = 3e-4,
        n_steps:         int   = 2048,
        batch_size:      int   = 64,
        n_epochs:        int   = 10,
        gamma:           float = 0.99,
        gae_lambda:      float = 0.95,
        clip_range:      float = 0.2,
        ent_coef:        float = 0.0,
        vf_coef:         float = 0.5,
        max_grad_norm:   float = 0.5,
        seed:            int   = 0,
        tensorboard_log: str | None = None,
        verbose:         int   = 1,
        **kwargs,
    ):
        super().__init__(env, seed=seed, tensorboard_log=tensorboard_log,
                         verbose=verbose, **kwargs)
        self.learning_rate = learning_rate
        self.n_steps       = n_steps
        self.batch_size    = batch_size
        self.n_epochs      = n_epochs
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_range    = clip_range
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm

        obs_dim   = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)

        self.policy_net = build_mlp(obs_dim, n_actions)
        self.value_net  = build_mlp(obs_dim, 1)
        # Last layer of value net: smaller init scale
        nn.init.orthogonal_(self.value_net[-1].weight, gain=1.0)

        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate, eps=1e-5,
        )

    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        with torch.no_grad():
            t = obs_to_tensor(obs).unsqueeze(0)
            logits = self.policy_net(t)
            dist   = Categorical(logits=logits)
            action = dist.mode if deterministic else dist.sample()
        return action.cpu().item(), None

    # ------------------------------------------------------------------

    def learn(self, total_timesteps: int, callback=None,
              progress_bar: bool = False, reset_num_timesteps: bool = True):
        if reset_num_timesteps:
            self.num_timesteps = 0

        cb = self._wrap_callback(callback)
        cb.init_callback(self)

        pbar = self._progress(total_timesteps) if progress_bar else None
        obs, _ = self.env.reset()

        while self.num_timesteps < total_timesteps:
            # ── collect n_steps ──────────────────────────────────────
            buf_obs, buf_act, buf_rew, buf_done = [], [], [], []
            buf_logp, buf_val = [], []

            for _ in range(self.n_steps):
                t_obs = obs_to_tensor(obs).unsqueeze(0)
                with torch.no_grad():
                    logits = self.policy_net(t_obs)
                    value  = self.value_net(t_obs).squeeze()
                    dist   = Categorical(logits=logits)
                    action = dist.sample()
                    logp   = dist.log_prob(action)

                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                buf_obs.append(obs)
                buf_act.append(action.item())
                buf_rew.append(float(reward))
                buf_done.append(float(done))
                buf_logp.append(logp.item())
                buf_val.append(value.item())

                obs = next_obs
                self.num_timesteps += 1
                if pbar:
                    pbar.update(1)

                if not cb.on_step():
                    if pbar:
                        pbar.close()
                    return self

                if done:
                    obs, _ = self.env.reset()

            # ── bootstrap last value ──────────────────────────────────
            with torch.no_grad():
                last_val = self.value_net(obs_to_tensor(obs).unsqueeze(0)).item()

            # ── GAE ───────────────────────────────────────────────────
            advantages = np.zeros(self.n_steps, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(self.n_steps)):
                next_val  = buf_val[t + 1] if t + 1 < self.n_steps else last_val
                next_done = buf_done[t]
                delta = (buf_rew[t]
                         + self.gamma * next_val * (1.0 - next_done)
                         - buf_val[t])
                gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
                advantages[t] = gae

            returns = advantages + np.array(buf_val, dtype=np.float32)

            # ── tensors ───────────────────────────────────────────────
            t_obs  = torch.tensor(np.array(buf_obs),  dtype=torch.float32, device=DEVICE)
            t_act  = torch.tensor(buf_act,             dtype=torch.long,    device=DEVICE)
            t_logp = torch.tensor(buf_logp,            dtype=torch.float32, device=DEVICE)
            t_adv  = torch.tensor(advantages,          dtype=torch.float32, device=DEVICE)
            t_ret  = torch.tensor(returns,             dtype=torch.float32, device=DEVICE)

            # ── PPO update ────────────────────────────────────────────
            idx = np.arange(self.n_steps)
            for _ in range(self.n_epochs):
                np.random.shuffle(idx)
                for start in range(0, self.n_steps, self.batch_size):
                    mb = idx[start: start + self.batch_size]

                    mb_obs  = t_obs[mb]
                    mb_act  = t_act[mb]
                    mb_logp = t_logp[mb]
                    mb_adv  = t_adv[mb]
                    mb_ret  = t_ret[mb]

                    # Normalize advantages per minibatch
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    logits  = self.policy_net(mb_obs)
                    dist    = Categorical(logits=logits)
                    new_logp = dist.log_prob(mb_act)
                    entropy  = dist.entropy().mean()

                    values  = self.value_net(mb_obs).squeeze(-1)

                    ratio   = torch.exp(new_logp - mb_logp)
                    clip_r  = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    policy_loss = -torch.min(ratio * mb_adv, clip_r * mb_adv).mean()
                    value_loss  = nn.functional.mse_loss(values, mb_ret)
                    loss = (policy_loss
                            + self.vf_coef   * value_loss
                            - self.ent_coef  * entropy)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.policy_net.parameters()) +
                        list(self.value_net.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()

            cb.on_rollout_end()

        if pbar:
            pbar.close()
        cb.on_training_end()
        return self

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _hyperparams(self) -> dict:
        return dict(
            learning_rate=self.learning_rate, n_steps=self.n_steps,
            batch_size=self.batch_size, n_epochs=self.n_epochs,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
            clip_range=self.clip_range, ent_coef=self.ent_coef,
            vf_coef=self.vf_coef, max_grad_norm=self.max_grad_norm,
            seed=self.seed, verbose=self.verbose,
        )

    def _state_dict(self) -> dict:
        return {
            "policy_net": self.policy_net.state_dict(),
            "value_net":  self.value_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }

    def _load_state_dict(self, sd: dict) -> None:
        self.policy_net.load_state_dict(sd["policy_net"])
        self.value_net.load_state_dict(sd["value_net"])
        self.optimizer.load_state_dict(sd["optimizer"])
