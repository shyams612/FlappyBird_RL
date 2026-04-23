"""Monitor wrapper — records episode reward/length/time to CSV."""
from __future__ import annotations
import time
import json
from pathlib import Path

import gymnasium
import numpy as np


class Monitor(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, filename: str | None = None):
        super().__init__(env)
        self._t_start = time.time()
        self._ep_reward = 0.0
        self._ep_length = 0
        self._file = None

        if filename is not None:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(filename, "w")
            self._file.write(json.dumps({"t_start": self._t_start}) + "\n")
            self._file.write("r,l,t\n")
            self._file.flush()

    def reset(self, **kwargs):
        self._ep_reward = 0.0
        self._ep_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward += float(reward)
        self._ep_length += 1
        if terminated or truncated:
            elapsed = time.time() - self._t_start
            if self._file is not None:
                self._file.write(f"{self._ep_reward:.6f},{self._ep_length},{elapsed:.6f}\n")
                self._file.flush()
            self._ep_reward = 0.0
            self._ep_length = 0
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        super().close()
