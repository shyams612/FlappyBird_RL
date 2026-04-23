"""DQN — Deep Q-Network (off-policy, replay buffer, epsilon-greedy, target network)."""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseAlgorithm, build_mlp, obs_to_tensor, DEVICE


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.pos      = 0
        self.full     = False
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos]      = obs
        self.actions[self.pos]  = action
        self.rewards[self.pos]  = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos]    = float(done)
        self.pos  = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int):
        n   = self.capacity if self.full else self.pos
        idx = np.random.randint(0, n, size=batch_size)
        return (
            torch.tensor(self.obs[idx],      device=DEVICE),
            torch.tensor(self.actions[idx],  device=DEVICE),
            torch.tensor(self.rewards[idx],  device=DEVICE),
            torch.tensor(self.next_obs[idx], device=DEVICE),
            torch.tensor(self.dones[idx],    device=DEVICE),
        )

    def __len__(self):
        return self.capacity if self.full else self.pos


class DQN(BaseAlgorithm):

    def __init__(
        self,
        env,
        learning_rate:         float = 1e-4,
        batch_size:            int   = 32,
        buffer_size:           int   = 1_000_000,
        learning_starts:       int   = 50_000,
        gamma:                 float = 0.99,
        exploration_fraction:  float = 0.1,
        exploration_final_eps: float = 0.05,
        target_update_interval:int   = 10_000,
        gradient_steps:        int   = 1,
        tau:                   float = 1.0,
        seed:                  int   = 0,
        tensorboard_log:       str | None = None,
        verbose:               int   = 1,
        **kwargs,
    ):
        super().__init__(env, seed=seed, tensorboard_log=tensorboard_log,
                         verbose=verbose, **kwargs)
        self.learning_rate          = learning_rate
        self.batch_size             = batch_size
        self.buffer_size            = buffer_size
        self.learning_starts        = learning_starts
        self.gamma                  = gamma
        self.exploration_fraction   = exploration_fraction
        self.exploration_final_eps  = exploration_final_eps
        self.target_update_interval = target_update_interval
        self.gradient_steps         = gradient_steps
        self.tau                    = tau

        obs_dim   = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)
        self._obs_dim   = obs_dim
        self._n_actions = n_actions

        self.q_net      = build_mlp(obs_dim, n_actions)
        self.q_target   = copy.deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate, eps=1e-5)
        self.buffer    = ReplayBuffer(buffer_size, obs_dim)
        self._epsilon  = 1.0

    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if not deterministic and np.random.random() < self._epsilon:
            return self.env.action_space.sample(), None
        with torch.no_grad():
            q = self.q_net(obs_to_tensor(obs).unsqueeze(0))
        return q.argmax(dim=1).item(), None

    # ------------------------------------------------------------------

    def _update_epsilon(self, total_timesteps: int):
        progress = min(self.num_timesteps / max(self.exploration_fraction * total_timesteps, 1), 1.0)
        self._epsilon = 1.0 + progress * (self.exploration_final_eps - 1.0)

    def _soft_update_target(self):
        for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

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
            self._update_epsilon(total_timesteps)

            # epsilon-greedy action
            if (self.num_timesteps < self.learning_starts
                    or np.random.random() < self._epsilon):
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    q = self.q_net(obs_to_tensor(obs).unsqueeze(0))
                action = q.argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs if not done else self.env.reset()[0]

            self.num_timesteps += 1
            if pbar:
                pbar.update(1)

            # ── gradient update ───────────────────────────────────────
            if (self.num_timesteps >= self.learning_starts
                    and len(self.buffer) >= self.batch_size):
                for _ in range(self.gradient_steps):
                    s, a, r, s2, d = self.buffer.sample(self.batch_size)

                    with torch.no_grad():
                        target_q = r + self.gamma * self.q_target(s2).max(dim=1).values * (1.0 - d)

                    current_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                    loss = nn.functional.mse_loss(current_q, target_q)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
                    self.optimizer.step()

            # ── target network update ─────────────────────────────────
            if self.num_timesteps % self.target_update_interval == 0:
                self._soft_update_target()

            if not cb.on_step():
                break

        if pbar:
            pbar.close()
        cb.on_training_end()
        return self

    # ------------------------------------------------------------------

    def _hyperparams(self) -> dict:
        return dict(
            learning_rate=self.learning_rate, batch_size=self.batch_size,
            buffer_size=self.buffer_size, learning_starts=self.learning_starts,
            gamma=self.gamma, exploration_fraction=self.exploration_fraction,
            exploration_final_eps=self.exploration_final_eps,
            target_update_interval=self.target_update_interval,
            gradient_steps=self.gradient_steps, tau=self.tau,
            seed=self.seed, verbose=self.verbose,
        )

    def _state_dict(self) -> dict:
        return {
            "q_net":     self.q_net.state_dict(),
            "q_target":  self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon":   self._epsilon,
            "obs_dim":   self._obs_dim,
            "n_actions": self._n_actions,
        }

    def _load_state_dict(self, sd: dict) -> None:
        self.q_net.load_state_dict(sd["q_net"])
        self.q_target.load_state_dict(sd["q_target"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self._epsilon = sd.get("epsilon", self.exploration_final_eps)
