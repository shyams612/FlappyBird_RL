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
    gap_height: float        = 160  # vertical opening size
    gap_y_min: float         = 80   # min y of gap top edge
    gap_y_max: float         = 360  # max y of gap top edge (keeps gap on screen)

    # ------------------------------------------------------------------ health
    health_start: float      = 100.0
    passive_drain: float     = 0.0   # hp/frame — set > 0 for Config 3+

    # ------------------------------------------------------------------- misc
    bird_x: float            = 150   # fixed horizontal position of the bird
    max_episode_frames: int  = 18000  # 5 minutes at 60fps — safety cap

    # ------------------------------------------------------------------ flags
    # These are all False in v1; flip them for Config 2+
    enable_bullets: bool      = False
    enable_wind: bool         = False
    enable_health_kits: bool  = False

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