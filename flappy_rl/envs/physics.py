# envs/physics.py
"""
Pure physics functions.  Both modes return a NEW Bird — no mutation.
GameState.step() selects which function to call via cfg.physics_mode.

Discrete mode  — classic Flappy Bird feel.
  vel is in pixels/frame.  gravity and flap_strength are per-frame constants.

Continuous mode — dt-based integration.
  vel is in pixels/second.  gravity is px/s², flap_strength is an
  instantaneous velocity impulse in px/s.  Pass dt (seconds) each call.
"""

from __future__ import annotations
from envs.entities import Bird, Vec2
from envs.config import EnvConfig, PhysicsMode


# ---------------------------------------------------------------------------
# Discrete  (one call = one frame)
# ---------------------------------------------------------------------------

def discrete_step(bird: Bird, action: int, cfg: EnvConfig) -> Bird:
    """
    action: 0 = do nothing, 1 = flap
    Returns new Bird with updated pos and vel.
    Boundary check: bird is killed if it exits the screen vertically.
    """
    b = bird.copy()

    if action == 1:
        b.vel.y = cfg.flap_strength          # upward impulse (negative y)

    b.vel.y += cfg.gravity                   # gravity pulls down each frame
    b.pos.y += b.vel.y

    # Screen boundary — floors and ceilings are lethal in v1
    if b.pos.y - b.height / 2 <= 0 or b.pos.y + b.height / 2 >= cfg.screen_h:
        b.alive = False

    return b


# ---------------------------------------------------------------------------
# Continuous  (dt-based integration)
# ---------------------------------------------------------------------------

def continuous_step(bird: Bird, action: int, cfg: EnvConfig, dt: float) -> Bird:
    """
    dt: elapsed seconds since last call (typically 1/fps ≈ 0.0167 s).
    gravity_continuous and flap_strength_continuous should be set in cfg
    when physics_mode == CONTINUOUS; falls back to scaled discrete values.
    """
    b = bird.copy()

    if action == 1:
        # Impulse: override vertical velocity instantly
        b.vel.y = cfg.flap_strength_continuous

    b.vel.y += cfg.gravity_continuous * dt   # integrate acceleration
    b.pos.y += b.vel.y * dt                  # integrate velocity

    if b.pos.y - b.height / 2 <= 0 or b.pos.y + b.height / 2 >= cfg.screen_h:
        b.alive = False

    return b


# ---------------------------------------------------------------------------
# Dispatcher — called by GameState so it never has to import mode directly
# ---------------------------------------------------------------------------

def step_bird(bird: Bird, action: int, cfg: EnvConfig, dt: float | None = None) -> Bird:
    if cfg.physics_mode == PhysicsMode.DISCRETE:
        return discrete_step(bird, action, cfg)
    return continuous_step(bird, action, cfg, dt if dt is not None else 1.0 / cfg.fps)