# envs/flappy_env.py
"""
FlappyBirdEnv — standard Gymnasium interface over GameState.

Construction:
    env = FlappyBirdEnv(cfg)                          # simple obs, survival reward, headless
    env = FlappyBirdEnv(cfg, render_mode="human")     # with Pygame window
    env = FlappyBirdEnv(cfg,
                        obs_builder=LidarObsBuilder(cfg),
                        reward_fn=ScoredReward())      # swap obs / reward freely

The env knows nothing about Pygame directly — it delegates all rendering
to Renderer, which is only instantiated when render_mode="human".
"""

from __future__ import annotations
from typing import Any

import numpy as np
import gymnasium
from gymnasium import spaces

from envs.config import EnvConfig
from envs.game_state import GameState
from envs.observations import ObservationBuilder, SimpleObsBuilder
from envs.rewards import RewardFn, SurvivalReward


class FlappyBirdEnv(gymnasium.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        cfg: EnvConfig | None = None,
        obs_builder: ObservationBuilder | None = None,
        reward_fn: RewardFn | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.cfg          = cfg or EnvConfig()
        self.obs_builder  = obs_builder or SimpleObsBuilder(self.cfg)
        self.reward_fn    = reward_fn   or SurvivalReward()
        self.render_mode  = render_mode

        # Gymnasium spaces — derived entirely from the obs_builder
        self.observation_space = self.obs_builder.observation_space()
        self.action_space      = spaces.Discrete(2)   # 0 = nothing, 1 = flap

        # Internal state (set properly in reset())
        self._state: GameState | None = None

        # Renderer — only created on first render() call if render_mode=="human"
        self._renderer = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Use provided seed once; after that use None so each episode
        # gets a different random layout (no memorisation of a single track).
        episode_seed = seed
        self.obs_builder.reset()                        # clear frame stacks etc.
        self._state = GameState.reset(self.cfg, seed=episode_seed)
        obs = self.obs_builder.build(self._state)
        return obs, self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "Call reset() before step()."

        prev_state  = self._state
        self._state = self._state.step(int(action))

        obs        = self.obs_builder.build(self._state)
        reward     = self.reward_fn(prev_state, self._state, action)
        terminated = self._state.terminated
        truncated  = self._state.truncated

        return obs, reward, terminated, truncated, self._info()

    def render(self) -> None:
        if self.render_mode != "human":
            return
        self._ensure_renderer()
        self._renderer.draw(self._state)

    def close(self) -> None:
        if self._renderer is not None:
            import pygame
            pygame.quit()
            self._renderer = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _info(self) -> dict[str, Any]:
        if self._state is None:
            return {}
        return {
            "score":  self._state.score,
            "health": self._state.health,
            "frame":  self._state.frame,
        }

    def _ensure_renderer(self) -> None:
        if self._renderer is None:
            import pygame
            pygame.init()
            from rendering.renderer import Renderer
            self._renderer = Renderer(self.cfg.screen_w, self.cfg.screen_h)