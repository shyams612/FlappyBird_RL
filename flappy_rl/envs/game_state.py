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

from envs.entities import Bird, Pipe, PipeType, PIPE_DAMAGE, PIPE_DAMAGE_RANGE, Vec2
from envs.config import EnvConfig
from envs.physics import step_bird
from envs.spawner import PipeSpawner


def _gradient_damage(bird: Bird, pipe: Pipe) -> float:
    """
    Compute gradient damage for a non-hard pipe collision.

    Damage is highest near the gap edge and lowest far from the gap (near ceiling/floor).
    No instant death — max damage is capped at the pipe type's max value.

    This creates emergent health-aware behavior:
      High health → crash through pipe far from gap (low damage), skip precision flying
      Low health  → must thread gap carefully OR seek health kit first

    Uses penetration depth from the gap edge to determine position within the pipe:
      - Small penetration (bird center near gap, clipping edge) → near gap → max damage
      - Large penetration (bird deep in pipe body, far from gap) → min damage

    t = 0.0 → at gap edge   → max_damage
    t = 1.0 → at far edge   → min_damage
    """
    min_dmg, max_dmg = PIPE_DAMAGE_RANGE[pipe.pipe_type]

    bird_top = bird.pos.y - bird.height / 2
    bird_bot = bird.pos.y + bird.height / 2

    # Penetration depth from gap edge into the pipe body
    top_penetration = max(0.0, pipe.gap_top  - bird_top)   # clipping top pipe from below
    bot_penetration = max(0.0, bird_bot - pipe.gap_bottom)  # clipping bottom pipe from above
    penetration = max(top_penetration, bot_penetration)

    # Normalise against a reference depth — beyond this the bird is "deep" in the pipe
    # Use the full pipe segment height as reference so t is consistent across gap positions
    if bird_top < pipe.gap_top:
        pipe_thickness = max(pipe.gap_top, 1.0)
    else:
        pipe_thickness = max(9999.0 - pipe.gap_bottom, 1.0)

    t = min(penetration / pipe_thickness, 1.0)

    # t=0 (near gap) → max_dmg,  t=1 (far from gap) → min_dmg
    return max_dmg + t * (min_dmg - max_dmg)


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

        # 1. Advance bird physics; tick down invincibility
        new_bird = step_bird(self.bird, action, cfg, dt)
        new_bird.invincibility_frames = max(0, self.bird.invincibility_frames - 1)

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

            # Collision detection
            if new_bird.alive and not np.shattered:
                bird_rect = new_bird.rect
                if bird_rect.overlaps(np.top_rect) or bird_rect.overlaps(np.bot_rect):
                    if np.pipe_type == PipeType.HARD:
                        # HARD pipe — instant death
                        new_bird.alive = False
                    else:
                        # Non-hard pipe — never instant death, always survivable if health allows
                        if cfg.enable_gradient_damage:
                            damage = _gradient_damage(new_bird, np)
                        else:
                            damage = PIPE_DAMAGE[np.pipe_type]  # flat damage

                        if new_bird.invincibility_frames == 0:
                            new_health -= damage
                            if new_health <= 0:
                                new_bird.alive = False
                            else:
                                np.shattered = True
                                new_bird.invincibility_frames = cfg.invincibility_duration

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