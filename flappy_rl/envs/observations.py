# envs/observations.py
"""
ObservationBuilder abstraction.

To add a new observation type:
  1. Subclass ObservationBuilder
  2. Implement build() and observation_space()
  3. Pass it into FlappyBirdEnv at construction — nothing else changes.

SimpleObsBuilder (v1)
---------------------
8-dimensional hand-crafted vector. Fast to train on, easy to inspect.

  [0] bird_y              — normalised 0..1  (0 = top, 1 = bottom)
  [1] bird_vel_y          — normalised -1..1  (clipped)
  [2] next_pipe_dist_x    — normalised 0..1  (distance to next pipe's left edge)
  [3] next_pipe_gap_top   — normalised 0..1
  [4] next_pipe_gap_bot   — normalised 0..1
  [5] next2_pipe_dist_x   — second upcoming pipe, same encoding (0 if none)
  [6] next2_pipe_gap_top  — second upcoming pipe
  [7] health              — normalised 0..1

LidarObsBuilder (stub — Config 2+)
-----------------------------------
Placeholder that raises NotImplementedError so the import path exists
and can be wired in without changing FlappyBirdEnv.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import gymnasium
from gymnasium import spaces

from envs.game_state import GameState
from envs.config import EnvConfig


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class ObservationBuilder(ABC):

    @abstractmethod
    def build(self, state: GameState) -> np.ndarray:
        """Convert a GameState into a flat float32 observation vector."""

    @abstractmethod
    def observation_space(self) -> spaces.Box:
        """Return the corresponding Gymnasium observation space."""

    def reset(self) -> None:
        """Called on env.reset() — override for stateful builders (e.g. frame stack)."""


# ---------------------------------------------------------------------------
# SimpleObsBuilder
# ---------------------------------------------------------------------------

class SimpleObsBuilder(ObservationBuilder):
    """
    8-dim normalised vector.  No frame stacking, no LiDAR.
    All values are clipped to [-1, 1] or [0, 1] so the network
    receives well-scaled inputs from day one.
    """

    OBS_DIM = 8

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._vel_scale = 20.0   # rough max |vel_y| in pixels/frame

    # ------------------------------------------------------------------

    def build(self, state: GameState) -> np.ndarray:
        cfg = self.cfg
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Bird
        obs[0] = np.clip(state.bird.pos.y / cfg.screen_h, 0.0, 1.0)
        obs[1] = np.clip(state.bird.vel.y / self._vel_scale, -1.0, 1.0)

        # Upcoming pipes — find the next two that haven't been passed yet
        # and are still ahead of the bird
        upcoming = [
            p for p in state.pipes
            if p.x + p.width > state.bird.pos.x
        ]
        upcoming.sort(key=lambda p: p.x)

        for i, slot in enumerate([(2, 3, 4), (5, 6, 7)]):
            dist_idx, top_idx, bot_idx = slot
            if i < len(upcoming):
                p = upcoming[i]
                dist = (p.x - state.bird.pos.x) / cfg.screen_w
                obs[dist_idx] = np.clip(dist, 0.0, 1.0)
                obs[top_idx]  = np.clip(p.gap_top    / cfg.screen_h, 0.0, 1.0)
                obs[bot_idx]  = np.clip(p.gap_bottom / cfg.screen_h, 0.0, 1.0)
            else:
                # No second pipe visible yet — safe default
                obs[dist_idx] = 1.0
                obs[top_idx]  = 0.3
                obs[bot_idx]  = 0.7

        # Health
        obs[7] = np.clip(state.health / 100.0, 0.0, 1.0)

        return obs

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low  = np.full(self.OBS_DIM, -1.0, dtype=np.float32),
            high = np.full(self.OBS_DIM,  1.0, dtype=np.float32),
            dtype= np.float32,
        )


# ---------------------------------------------------------------------------
# LidarObsBuilder  (stub)
# ---------------------------------------------------------------------------

class LidarObsBuilder(ObservationBuilder):
    """
    Full LiDAR observation as described in the proposal:
      16 rays × (distance + type) + velocity + health, stacked × 4 frames
      → 136-dim flat float32 vector.

    Not implemented yet — raises NotImplementedError on use.
    Swap into FlappyBirdEnv when ready:
        env = FlappyBirdEnv(cfg, obs_builder=LidarObsBuilder(cfg))
    """

    N_RAYS       = 16
    STACK_FRAMES = 4
    FRAME_DIM    = N_RAYS * 2 + 2   # distances + types + vel + health = 34
    OBS_DIM      = FRAME_DIM * STACK_FRAMES  # 136

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    def build(self, state: GameState) -> np.ndarray:
        raise NotImplementedError("LidarObsBuilder not yet implemented.")

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )