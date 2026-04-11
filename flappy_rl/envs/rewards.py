# envs/rewards.py
"""
RewardFn abstraction.

To add a new reward signal:
  1. Subclass RewardFn
  2. Implement __call__()
  3. Pass it into FlappyBirdEnv — nothing else changes.

Available:
  SurvivalReward   — +1/frame alive, −100 on death  (default, v1)
  ScoredReward     — SurvivalReward + bonus per pipe passed
  HealthAwareReward — placeholder for Config 3+ experiments
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from envs.game_state import GameState


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class RewardFn(ABC):

    @abstractmethod
    def __call__(
        self,
        prev: GameState,
        curr: GameState,
        action: int,
    ) -> float:
        """
        Compute scalar reward for the transition prev → curr.

        prev:   state before step()
        curr:   state after  step()
        action: action taken (0 or 1)
        """


# ---------------------------------------------------------------------------
# SurvivalReward  (default v1)
# ---------------------------------------------------------------------------

class SurvivalReward(RewardFn):
    """
    +1  every frame the bird is alive.
    −100 on death.

    This is the simplest possible signal — the agent learns to survive
    longer purely through the accumulated +1s.  No explicit encouragement
    to pass pipes; that emerges from staying alive.
    """

    def __init__(self, death_penalty: float = -100.0):
        self.death_penalty = death_penalty

    def __call__(self, prev: GameState, curr: GameState, action: int) -> float:
        if curr.terminated:
            return self.death_penalty
        return 1.0


# ---------------------------------------------------------------------------
# ScoredReward
# ---------------------------------------------------------------------------

class ScoredReward(RewardFn):
    """
    SurvivalReward + a bonus each time the bird passes a pipe.

    pipe_bonus=10 means passing a pipe is worth 10 frames of survival.
    Good for speeding up early learning if the agent is stuck not passing
    any pipes at all.
    """

    def __init__(self, death_penalty: float = -100.0, pipe_bonus: float = 10.0):
        self.death_penalty = death_penalty
        self.pipe_bonus    = pipe_bonus

    def __call__(self, prev: GameState, curr: GameState, action: int) -> float:
        if curr.terminated:
            return self.death_penalty
        reward = 1.0
        if curr.score > prev.score:
            reward += self.pipe_bonus * (curr.score - prev.score)
        return reward


# ---------------------------------------------------------------------------
# HealthAwareReward  (stub — Config 3+)
# ---------------------------------------------------------------------------

class HealthAwareReward(RewardFn):
    """
    Placeholder for the health-management reward experiment described in
    the proposal.  Raises NotImplementedError until implemented.

    Intended signal:
      +1/frame alive
      + small bonus for collecting health kits  (or implicit via survival)
      − penalty proportional to unnecessary damage taken
      − death penalty
    """

    def __call__(self, prev: GameState, curr: GameState, action: int) -> float:
        raise NotImplementedError("HealthAwareReward not yet implemented.")