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
from envs.entities import PipeType


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
# Config2ObsBuilder
# ---------------------------------------------------------------------------

class Config2ObsBuilder(ObservationBuilder):
    """
    10-dimensional observation for Config 2 (pipe variants enabled).

    Extends SimpleObsBuilder by adding the pipe type for the next two
    upcoming pipes, so the agent can distinguish HARD/SOFT/BRITTLE/FOAM
    and adjust its risk tolerance accordingly.

      [0]  bird_y              — normalised 0..1
      [1]  bird_vel_y          — normalised -1..1
      [2]  next pipe dist_x    — normalised 0..1
      [3]  next pipe gap_top   — normalised 0..1
      [4]  next pipe gap_bot   — normalised 0..1
      [5]  next pipe type      — 0=HARD, 0.33=SOFT, 0.67=BRITTLE, 1.0=FOAM
      [6]  2nd pipe dist_x     — normalised 0..1
      [7]  2nd pipe gap_top    — normalised 0..1
      [8]  2nd pipe gap_bot    — normalised 0..1
      [9]  2nd pipe type       — same encoding
      [10] health              — normalised 0..1
    """

    OBS_DIM    = 11
    _TYPE_NORM = 3.0   # max PipeType int value (FOAM=3) — normalises to [0, 1]

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._vel_scale = 20.0

    def build(self, state: GameState) -> np.ndarray:
        cfg = self.cfg
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Bird
        obs[0] = np.clip(state.bird.pos.y / cfg.screen_h, 0.0, 1.0)
        obs[1] = np.clip(state.bird.vel.y / self._vel_scale, -1.0, 1.0)

        # Next two upcoming pipes
        upcoming = [p for p in state.pipes if p.x + p.width > state.bird.pos.x]
        upcoming.sort(key=lambda p: p.x)

        slots = [(2, 3, 4, 5), (6, 7, 8, 9)]
        for i, (dist_i, top_i, bot_i, type_i) in enumerate(slots):
            if i < len(upcoming):
                p = upcoming[i]
                obs[dist_i] = np.clip((p.x - state.bird.pos.x) / cfg.screen_w, 0.0, 1.0)
                obs[top_i]  = np.clip(p.gap_top    / cfg.screen_h, 0.0, 1.0)
                obs[bot_i]  = np.clip(p.gap_bottom / cfg.screen_h, 0.0, 1.0)
                obs[type_i] = float(p.pipe_type) / self._TYPE_NORM
            else:
                obs[dist_i] = 1.0
                obs[top_i]  = 0.3
                obs[bot_i]  = 0.7
                obs[type_i] = 0.0   # default to HARD encoding

        # Health
        obs[10] = np.clip(state.health / 100.0, 0.0, 1.0)

        return obs

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low  = np.full(self.OBS_DIM, -1.0, dtype=np.float32),
            high = np.full(self.OBS_DIM,  1.0, dtype=np.float32),
            dtype= np.float32,
        )


# ---------------------------------------------------------------------------
# Config2NoisyObsBuilder
# ---------------------------------------------------------------------------

class Config2NoisyObsBuilder(ObservationBuilder):
    """
    Same 11-dim layout as Config2ObsBuilder, but gap_top and gap_bottom
    are reported with Gaussian noise for non-HARD pipes.

    Motivation: the agent cannot precisely locate the gap edge of a soft/
    brittle/foam pipe — it only knows approximately where it is safe to fly.
    This forces genuine risk-reward tradeoffs:
      - At high health, the agent can afford to guess wrong and clip a FOAM pipe.
      - At low health, uncertainty makes any non-hard pipe feel dangerous.

    Noise is scaled inversely by damage: lower-damage pipes are noisier
    because the cost of a wrong guess is lower, making exploration worthwhile.

    Noise std per pipe type (in normalised units, i.e. fraction of screen_h):
      HARD    — 0.0   (exact — the agent knows exactly where death is)
      SOFT    — noise_soft
      BRITTLE — noise_brittle
      FOAM    — noise_foam   (noisiest — cheapest to be wrong about)

    Defaults (tunable):
      noise_foam    = 0.08   (~48 px on 600px screen)
      noise_soft    = 0.05   (~30 px)
      noise_brittle = 0.02   (~12 px)

    Layout (same as Config2ObsBuilder):
      [0]  bird_y
      [1]  bird_vel_y
      [2]  next pipe dist_x
      [3]  next pipe gap_top    ← noisy for non-HARD
      [4]  next pipe gap_bot    ← noisy for non-HARD
      [5]  next pipe type
      [6]  2nd pipe dist_x
      [7]  2nd pipe gap_top     ← noisy for non-HARD
      [8]  2nd pipe gap_bot     ← noisy for non-HARD
      [9]  2nd pipe type
      [10] health
    """

    OBS_DIM    = 11
    _TYPE_NORM = 3.0

    # Noise std in normalised units (fraction of screen_h) per pipe type
    _NOISE_STD: dict[PipeType, float] = {
        PipeType.HARD:    0.00,
        PipeType.SOFT:    0.05,
        PipeType.BRITTLE: 0.02,
        PipeType.FOAM:    0.08,
    }

    def __init__(
        self,
        cfg: EnvConfig,
        noise_foam:    float = 0.08,
        noise_soft:    float = 0.05,
        noise_brittle: float = 0.02,
        seed: int | None = None,
    ):
        self.cfg  = cfg
        self._vel_scale = 20.0
        self._rng = np.random.default_rng(seed)
        self._noise_std = {
            PipeType.HARD:    0.00,
            PipeType.SOFT:    noise_soft,
            PipeType.BRITTLE: noise_brittle,
            PipeType.FOAM:    noise_foam,
        }

    def reset(self) -> None:
        # Do not re-seed on reset — keeps noise behaviour consistent across episodes
        pass

    def build(self, state: GameState) -> np.ndarray:
        cfg = self.cfg
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Bird
        obs[0] = np.clip(state.bird.pos.y / cfg.screen_h, 0.0, 1.0)
        obs[1] = np.clip(state.bird.vel.y / self._vel_scale, -1.0, 1.0)

        # Next two upcoming pipes
        upcoming = [p for p in state.pipes if p.x + p.width > state.bird.pos.x]
        upcoming.sort(key=lambda p: p.x)

        slots = [(2, 3, 4, 5), (6, 7, 8, 9)]
        for i, (dist_i, top_i, bot_i, type_i) in enumerate(slots):
            if i < len(upcoming):
                p    = upcoming[i]
                std  = self._noise_std[p.pipe_type]
                noise_top = self._rng.normal(0, std) if std > 0 else 0.0
                noise_bot = self._rng.normal(0, std) if std > 0 else 0.0

                obs[dist_i] = np.clip((p.x - state.bird.pos.x) / cfg.screen_w, 0.0, 1.0)
                obs[top_i]  = np.clip(p.gap_top    / cfg.screen_h + noise_top, 0.0, 1.0)
                obs[bot_i]  = np.clip(p.gap_bottom / cfg.screen_h + noise_bot, 0.0, 1.0)
                obs[type_i] = float(p.pipe_type) / self._TYPE_NORM
            else:
                obs[dist_i] = 1.0
                obs[top_i]  = 0.3
                obs[bot_i]  = 0.7
                obs[type_i] = 0.0

        # Health
        obs[10] = np.clip(state.health / 100.0, 0.0, 1.0)

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