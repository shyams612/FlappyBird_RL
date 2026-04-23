"""A2C — Advantage Actor-Critic (on-policy, single update per rollout, no clip)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .base import BaseAlgorithm, build_mlp, obs_to_tensor, DEVICE


class A2C(BaseAlgorithm):

    def __init__(
        self,
        env,
        learning_rate:   float = 7e-4,
        n_steps:         int   = 5,
        gamma:           float = 0.99,
        gae_lambda:      float = 1.0,
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
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm

        obs_dim   = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)

        self.policy_net = build_mlp(obs_dim, n_actions)
        self.value_net  = build_mlp(obs_dim, 1)
        nn.init.orthogonal_(self.value_net[-1].weight, gain=1.0)

        self.optimizer = optim.RMSprop(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate, eps=1e-5, alpha=0.99,
        )

    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        with torch.no_grad():
            t      = obs_to_tensor(obs).unsqueeze(0)
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
            buf_logp, buf_val, buf_ent = [], [], []

            for _ in range(self.n_steps):
                t_obs = obs_to_tensor(obs).unsqueeze(0)
                with torch.no_grad():
                    logits = self.policy_net(t_obs)
                    value  = self.value_net(t_obs).squeeze()
                    dist   = Categorical(logits=logits)
                    action = dist.sample()
                    logp   = dist.log_prob(action)
                    ent    = dist.entropy()

                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                buf_obs.append(obs)
                buf_act.append(action.item())
                buf_rew.append(float(reward))
                buf_done.append(float(done))
                buf_logp.append(logp.item())
                buf_val.append(value.item())
                buf_ent.append(ent.item())

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

            # ── single update (no minibatches, no epochs) ─────────────
            t_obs  = torch.tensor(np.array(buf_obs),  dtype=torch.float32, device=DEVICE)
            t_act  = torch.tensor(buf_act,             dtype=torch.long,    device=DEVICE)
            t_adv  = torch.tensor(advantages,          dtype=torch.float32, device=DEVICE)
            t_ret  = torch.tensor(returns,             dtype=torch.float32, device=DEVICE)

            t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

            logits  = self.policy_net(t_obs)
            dist    = Categorical(logits=logits)
            logp    = dist.log_prob(t_act)
            entropy = dist.entropy().mean()
            values  = self.value_net(t_obs).squeeze(-1)

            policy_loss = -(logp * t_adv).mean()
            value_loss  = nn.functional.mse_loss(values, t_ret)
            loss = (policy_loss
                    + self.vf_coef  * value_loss
                    - self.ent_coef * entropy)

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

    def _hyperparams(self) -> dict:
        return dict(
            learning_rate=self.learning_rate, n_steps=self.n_steps,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef, vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
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
