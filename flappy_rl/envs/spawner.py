# envs/spawner.py
"""
PipeSpawner: decides when and where to spawn new pipe columns.

Design:
  - Stateless with respect to GameState — it only tracks the x position
    of the next pipe to spawn.
  - Accepts an RNG instance so training envs can be seeded reproducibly.
  - tick() is a pure function: takes the current pipe list, returns an
    updated list + a new spawner (immutable update style).
"""

from __future__ import annotations
import random
from dataclasses import dataclass

from envs.entities import Pipe, PipeType
from envs.config import EnvConfig


@dataclass
class PipeSpawner:
    cfg: EnvConfig
    rng: random.Random
    next_pipe_x: float = 0.0   # world-x at which the next pipe should appear

    def __post_init__(self):
        if self.next_pipe_x == 0.0:
            self.next_pipe_x = self.cfg.screen_w + 60  # first pipe just off-screen

    # ------------------------------------------------------------------
    # Initial population
    # ------------------------------------------------------------------

    def initial_pipes(self) -> list[Pipe]:
        """
        Pre-populate enough pipes to fill the screen at episode start,
        so the bird isn't flying in empty space on frame 1.
        """
        pipes: list[Pipe] = []
        x = self.cfg.screen_w + 60
        # Fill screen width + a buffer
        while x < self.cfg.screen_w * 3:
            pipes.append(self._make_pipe(x))
            x += self.cfg.pipe_spacing
        self.next_pipe_x = x
        return pipes

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def tick(self, pipes: list[Pipe]) -> tuple[list[Pipe], PipeSpawner]:
        """
        Called once per frame after pipes have been scrolled.
        Appends a new pipe if the rightmost existing pipe has scrolled
        far enough left to need a successor.

        Returns: (updated pipe list, new PipeSpawner with updated state)
        """
        new_spawner = PipeSpawner(self.cfg, self.rng, self.next_pipe_x)
        new_pipes = list(pipes)

        # The rightmost pipe's right edge
        rightmost_x = max((p.x + p.width for p in pipes), default=0)

        if rightmost_x < self.cfg.screen_w + self.cfg.pipe_spacing:
            spawn_x = rightmost_x + self.cfg.pipe_spacing
            new_pipes.append(self._make_pipe(spawn_x))
            new_spawner.next_pipe_x = spawn_x + self.cfg.pipe_spacing

        return new_pipes, new_spawner

    # ------------------------------------------------------------------
    # Pipe factory
    # ------------------------------------------------------------------

    def _make_pipe(self, x: float) -> Pipe:
        pipe_type  = self._sample_pipe_type()
        gap_height = self._gap_height_for(pipe_type)
        gap_top    = self.rng.uniform(self.cfg.gap_y_min, self.cfg.gap_y_max)
        gap_bottom = gap_top + gap_height
        return Pipe(
            x=x,
            gap_top=gap_top,
            gap_bottom=gap_bottom,
            pipe_type=pipe_type,
        )

    def _gap_height_for(self, pipe_type: PipeType) -> float:
        """Return the gap height for the given pipe type."""
        if pipe_type == PipeType.SOFT:
            return self.cfg.gap_height_soft
        if pipe_type == PipeType.BRITTLE:
            return self.cfg.gap_height_brittle
        if pipe_type == PipeType.FOAM:
            return self.cfg.gap_height_foam
        return self.cfg.gap_height   # HARD — normal gap

    def _sample_pipe_type(self) -> PipeType:
        """
        When enable_pipe_variants is False, always return HARD.
        When True, sample from the four types using the configured weights.
        """
        if not self.cfg.enable_pipe_variants:
            return PipeType.HARD

        weights = [
            self.cfg.pipe_weight_hard,
            self.cfg.pipe_weight_soft,
            self.cfg.pipe_weight_brittle,
            self.cfg.pipe_weight_foam,
        ]
        types = [PipeType.HARD, PipeType.SOFT, PipeType.BRITTLE, PipeType.FOAM]
        return self.rng.choices(types, weights=weights, k=1)[0]