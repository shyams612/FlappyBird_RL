# envs/game_state.py
"""
GameState: the complete, self-contained description of the world at one moment.

Rules:
  - No Pygame imports.
  - No Gymnasium imports.
  - step() returns a NEW GameState; the original is never mutated.
  - reset() is a classmethod that produces a clean starting state.

Collision model (v1):
  - Bird rect ∩ pipe top/bot rect  →  instant death (hard pipe only).
  - Bird exits screen vertically    →  instant death (handled in physics).
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field

from envs.entities import Bird, Pipe, PipeType, Vec2
from envs.config import EnvConfig
from envs.physics import step_bird
from envs.spawner import PipeSpawner


@dataclass
class GameState:
    bird: Bird
    pipes: list[Pipe]
    spawner: PipeSpawner
    cfg: EnvConfig
    frame: int      = 0
    score: int      = 0     # pipes fully passed
    health: float   = 100.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls, cfg: EnvConfig, seed: int | None = None) -> GameState:
        rng = random.Random(seed)
        bird = Bird(
            pos=Vec2(cfg.bird_x, cfg.screen_h / 2),
            vel=Vec2(0.0, 0.0),
        )
        spawner = PipeSpawner(cfg, rng)
        pipes = spawner.initial_pipes()
        return cls(
            bird=bird,
            pipes=pipes,
            spawner=spawner,
            cfg=cfg,
            frame=0,
            score=0,
            health=cfg.health_start,
        )

    # ------------------------------------------------------------------
    # Main transition
    # ------------------------------------------------------------------

    def step(self, action: int, dt: float | None = None) -> GameState:
        """
        Advance the world by one frame.
        Returns a new GameState — does NOT mutate self.

        action: 0 = do nothing, 1 = flap
        dt:     elapsed seconds; only used in CONTINUOUS physics mode.
        """
        cfg = self.cfg

        # 1. Advance bird physics
        new_bird = step_bird(self.bird, action, cfg, dt)

        # 2. Passive health drain
        new_health = self.health - cfg.passive_drain
        if new_health <= 0:
            new_bird.alive = False

        # 3. Scroll and update pipes
        scroll = cfg.scroll_speed
        new_pipes: list[Pipe] = []
        new_score = self.score

        for p in self.pipes:
            np = p.copy()
            np.x -= scroll

            # Score: bird has fully passed this pipe column
            if not np.passed and new_bird.pos.x > np.x + np.width:
                np.passed = True
                new_score += 1

            # Collision detection (v1: hard pipes only → instant death)
            if new_bird.alive:
                bird_rect = new_bird.rect
                if bird_rect.overlaps(np.top_rect) or bird_rect.overlaps(np.bot_rect):
                    new_bird.alive = False

            # Cull pipes that have scrolled fully off the left edge
            if np.x + np.width > -10:
                new_pipes.append(np)

        # 4. Spawn new pipes as needed
        new_pipes, new_spawner = self.spawner.tick(new_pipes)

        return GameState(
            bird=new_bird,
            pipes=new_pipes,
            spawner=new_spawner,
            cfg=cfg,
            frame=self.frame + 1,
            score=new_score,
            health=max(0.0, new_health),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self.bird.alive

    @property
    def terminated(self) -> bool:
        """Episode is over (death or health depleted)."""
        return not self.alive

    @property
    def truncated(self) -> bool:
        """Hit the safety frame cap."""
        return self.frame >= self.cfg.max_episode_frames