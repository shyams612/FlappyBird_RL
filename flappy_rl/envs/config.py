# envs/config.py
"""
Single source of truth for every tunable constant.
Load from YAML via EnvConfig.from_yaml(), or construct directly in tests.

v1 defaults: 800x600, hard pipes only, discrete physics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml


class PhysicsMode(str, Enum):
    DISCRETE   = "discrete"
    CONTINUOUS = "continuous"


@dataclass
class EnvConfig:
    # ------------------------------------------------------------------ screen
    screen_w: int   = 800
    screen_h: int   = 600
    fps: int        = 60

    # ---------------------------------------------------------------- physics
    physics_mode: PhysicsMode = PhysicsMode.DISCRETE

    # Discrete physics  (units: pixels / frame)
    gravity: float        = 0.5
    flap_strength: float  = -5.0   # negative = upward

    # Continuous physics  (units: pixels / second)
    # Scaled from discrete defaults so both modes feel roughly equivalent:
    #   gravity_continuous  ≈ gravity * fps²  / some_factor
    #   flap_strength_continuous ≈ flap_strength * fps
    gravity_continuous:       float = 900.0
    flap_strength_continuous: float = -480.0

    # --------------------------------------------------------------- scrolling
    scroll_speed: float = 3.0      # pixels/frame (discrete) or px/s (continuous)

    # ------------------------------------------------------------------ pipes
    pipe_width: float        = 60
    pipe_spacing: float      = 280  # horizontal gap between pipe columns
    gap_height: float        = 160  # vertical opening for HARD pipes
    gap_y_min: float         = 80   # min y of gap top edge
    gap_y_max: float         = 360  # max y of gap top edge (keeps gap on screen)

    # Per-type gap heights for non-hard pipes (tighter than HARD to force precision)
    # Only used when enable_pipe_variants=True
    gap_height_soft:    float = 100.0
    gap_height_brittle: float =  90.0
    gap_height_foam:    float = 110.0

    # ------------------------------------------------------------------ health
    health_start: float      = 100.0
    passive_drain: float     = 0.0   # hp/frame — set > 0 for Config 3+

    # ------------------------------------------------------------------- misc
    bird_x: float            = 150   # fixed horizontal position of the bird
    max_episode_frames: int  = 18000  # 5 minutes at 60fps — safety cap

    # ------------------------------------------------------------------ flags
    # These are all False in v1; flip them for Config 2+
    enable_bullets: bool      = False
    bullet_count: int         = 10    # bullets available per episode when enable_bullets=True
    enable_wind: bool         = False
    enable_health_kits: bool  = False

    # --------------------------------------------------------- pipe variants
    # When False: all pipes are HARD (instant death) — Config 1 behaviour.
    # When True:  pipe type is sampled per column using the weights below.
    enable_pipe_variants: bool  = False
    pipe_weight_hard:    float  = 0.7   # proportional; normalised at spawn time
    pipe_weight_soft:    float  = 0.1   # SOFT    — 10 hp damage, bird passes through
    pipe_weight_brittle: float  = 0.1   # BRITTLE — 25 hp damage, bird passes through
    pipe_weight_foam:    float  = 0.1   # FOAM    —  5 hp damage, bird passes through

    # Frames of damage immunity granted after a non-hard pipe hit (prevents
    # health draining every frame the bird overlaps the pipe).
    invincibility_duration: int = 30    # 0.5 s at 60 fps

    # When True: damage scales with penetration depth (graze = small, deep crash = death)
    # When False: flat damage from PIPE_DAMAGE dict regardless of where bird hits
    enable_gradient_damage: bool = False

    # --------------------------------------------------------- health reward
    # null     → SurvivalReward (health ignored in reward)
    # continuous → HealthAwareReward (penalty scales continuously with health)
    # threshold  → ThresholdHealthReward (sharp penalty below health_reward_threshold)
    health_reward_fn: str | None  = None
    health_reward_scale: float    = 0.5    # damage_scale for continuous
    health_reward_threshold: float = 25.0  # hp below which threshold mode kicks in
    health_reward_threshold_scale: float = 3.0  # penalty multiplier below threshold

    # --------------------------------------------------------------- computed
    @property
    def first_pipe_x(self) -> float:
        """X position of the first pipe column at episode start."""
        return self.screen_w + 100  # spawn offscreen right

    # ----------------------------------------------------------- serialisation
    @classmethod
    def from_yaml(cls, path: str | Path) -> EnvConfig:
        raw = yaml.safe_load(Path(path).read_text())
        # Convert physics_mode string → enum
        if "physics_mode" in raw:
            raw["physics_mode"] = PhysicsMode(raw["physics_mode"])
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        d = {k: (v.value if isinstance(v, Enum) else v)
             for k, v in self.__dict__.items()}
        Path(path).write_text(yaml.dump(d, sort_keys=False))