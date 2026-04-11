# envs/entities.py
"""
Pure data — no logic, no Pygame, no imports beyond stdlib.
All game objects are defined here as dataclasses.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

@dataclass
class Vec2:
    x: float
    y: float

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def copy(self) -> Vec2:
        return Vec2(self.x, self.y)


@dataclass
class Rect:
    """Axis-aligned bounding box.  x, y = top-left corner."""
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:  return self.x + self.w
    @property
    def bottom(self) -> float: return self.y + self.h
    @property
    def cx(self) -> float:     return self.x + self.w / 2
    @property
    def cy(self) -> float:     return self.y + self.h / 2

    def overlaps(self, other: Rect) -> bool:
        return (
            self.x < other.right  and self.right  > other.x and
            self.y < other.bottom and self.bottom > other.y
        )

    def copy(self) -> Rect:
        return Rect(self.x, self.y, self.w, self.h)


# ---------------------------------------------------------------------------
# Pipe types  (v1: HARD only — kept as enum so later variants slot in cleanly)
# ---------------------------------------------------------------------------

class PipeType(IntEnum):
    HARD    = 0   # instant death
    SOFT    = 1   # -10 hp  (Config 2+)
    BRITTLE = 2   # -25 hp  (Config 2+)
    FOAM    = 3   # -5 hp   (Config 2+)


PIPE_DAMAGE: dict[PipeType, float] = {
    PipeType.HARD:    float("inf"),   # treated as instant death
    PipeType.SOFT:    10.0,
    PipeType.BRITTLE: 25.0,
    PipeType.FOAM:     5.0,
}


# ---------------------------------------------------------------------------
# Game entities
# ---------------------------------------------------------------------------

@dataclass
class Bird:
    pos: Vec2           # centre of the bird
    vel: Vec2           # pixels per frame (discrete) or per second (continuous)
    width: float  = 34
    height: float = 24
    alive: bool   = True
    health: float = 100.0

    @property
    def rect(self) -> Rect:
        return Rect(
            self.pos.x - self.width  / 2,
            self.pos.y - self.height / 2,
            self.width,
            self.height,
        )

    def copy(self) -> Bird:
        return Bird(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            width=self.width,
            height=self.height,
            alive=self.alive,
            health=self.health,
        )


@dataclass
class Pipe:
    """
    A pipe column.  x is the left edge in world-space (scrolls leftward).
    gap_top / gap_bottom define the open gap the bird must fly through.
    """
    x: float
    gap_top: float      # y coordinate of the top of the gap
    gap_bottom: float   # y coordinate of the bottom of the gap
    pipe_type: PipeType = PipeType.HARD
    width: float        = 60
    passed: bool        = False   # flipped once bird x > pipe x+width (for scoring)

    @property
    def top_rect(self) -> Rect:
        """Rect for the upper pipe segment."""
        return Rect(self.x, 0, self.width, self.gap_top)

    @property
    def bot_rect(self) -> Rect:
        """Rect for the lower pipe segment (gap_bottom → screen bottom)."""
        return Rect(self.x, self.gap_bottom, self.width, 9999)

    def copy(self) -> Pipe:
        return Pipe(
            x=self.x,
            gap_top=self.gap_top,
            gap_bottom=self.gap_bottom,
            pipe_type=self.pipe_type,
            width=self.width,
            passed=self.passed,
        )