# envs/rewards.py
"""
RewardFn abstraction.

To add a new reward signal:
  1. Subclass RewardFn
  2. Implement __call__()
  3. Pass it into FlappyBirdEnv — nothing else changes.

Available:
  SurvivalReward        — +1/frame alive, −100 on death  (default, v1)
  ScoredReward          — SurvivalReward + bonus per pipe passed
  HealthAwareReward     — continuous damage penalty scaling inversely with health
  ThresholdHealthReward — flat small penalty above threshold, sharp penalty below it

Health reward comparison (FOAM hit, 5hp damage):
  HealthAwareReward (scale=0.5):
    100hp → penalty 2.5   50hp → penalty 5.0   10hp → penalty 25.0
  ThresholdHealthReward (threshold=25hp, scale=0.5, threshold_scale=3.0):
    100hp → penalty 2.5   50hp → penalty 2.5   20hp → penalty 7.5 (3x multiplier kicks in)
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
# HealthAwareReward
# ---------------------------------------------------------------------------

class HealthAwareReward(RewardFn):
    """
    Reward that makes the agent increasingly risk-averse as health depletes.

    Signal per frame:
      +1                     — survival reward
      +pipe_bonus            — each time a pipe column is passed
      −damage_penalty        — on any damage taken, scaled by health urgency
      −death_penalty         — on death

    Damage penalty formula:
        penalty = damage_taken × (max_health / health_before_hit) × damage_scale

    At full health (100 hp):  penalty = damage × 1.0 × damage_scale  (small)
    At half health  (50 hp):  penalty = damage × 2.0 × damage_scale  (moderate)
    At low health   (10 hp):  penalty = damage × 10.0 × damage_scale (large)

    This creates a continuous risk-aversion gradient — the lower the health,
    the more costly any hit becomes, pushing the agent to treat non-hard pipes
    as increasingly dangerous as health depletes.
    """

    def __init__(
        self,
        death_penalty: float = -100.0,
        pipe_bonus:    float =   10.0,
        damage_scale:  float =    0.5,   # tune to balance against survival reward
        max_health:    float =  100.0,
    ):
        self.death_penalty = death_penalty
        self.pipe_bonus    = pipe_bonus
        self.damage_scale  = damage_scale
        self.max_health    = max_health

    def __call__(self, prev: GameState, curr: GameState, action: int) -> float:
        if curr.terminated:
            return self.death_penalty

        reward = 1.0

        # Pipe passing bonus
        if curr.score > prev.score:
            reward += self.pipe_bonus * (curr.score - prev.score)

        # Damage penalty — scaled by how critical health was before the hit
        damage_taken = prev.health - curr.health
        if damage_taken > 0:
            urgency = self.max_health / prev.health   # 1.0 at full health, → ∞ near zero
            reward -= damage_taken * urgency * self.damage_scale

        return reward


# ---------------------------------------------------------------------------
# ThresholdHealthReward
# ---------------------------------------------------------------------------

class ThresholdHealthReward(RewardFn):
    """
    Encodes an explicit mode switch: below a health threshold, the agent
    should treat non-hard pipes as near-fatal and avoid them entirely.

    Signal per frame:
      +1                 — survival reward
      +pipe_bonus        — each time a pipe column is passed
      −damage_penalty    — on any damage taken:
                           above threshold: flat penalty (damage × damage_scale)
                           below threshold: amplified penalty (damage × damage_scale × threshold_scale)
      −death_penalty     — on death

    Penalty at full health (100hp, FOAM 5hp hit, scale=0.5):
        above threshold → 5 × 0.5 = 2.5
    Penalty below threshold (20hp, FOAM 5hp hit, scale=0.5, threshold_scale=3.0):
        below threshold → 5 × 0.5 × 3.0 = 7.5

    The sharp jump at the threshold is the behavioral specification:
    "when health is critical, treat every hit as you would a hard pipe".
    """

    def __init__(
        self,
        death_penalty:     float = -100.0,
        pipe_bonus:        float =   10.0,
        damage_scale:      float =    0.5,
        threshold:         float =   25.0,   # hp below which multiplier kicks in
        threshold_scale:   float =    3.0,   # penalty multiplier below threshold
    ):
        self.death_penalty   = death_penalty
        self.pipe_bonus      = pipe_bonus
        self.damage_scale    = damage_scale
        self.threshold       = threshold
        self.threshold_scale = threshold_scale

    def __call__(self, prev: GameState, curr: GameState, action: int) -> float:
        if curr.terminated:
            return self.death_penalty

        reward = 1.0

        # Pipe passing bonus
        if curr.score > prev.score:
            reward += self.pipe_bonus * (curr.score - prev.score)

        # Damage penalty — mode switch at threshold
        damage_taken = prev.health - curr.health
        if damage_taken > 0:
            multiplier = self.threshold_scale if prev.health <= self.threshold else 1.0
            reward -= damage_taken * self.damage_scale * multiplier

        return reward


# ---------------------------------------------------------------------------
# ExponentialHealthReward
# ---------------------------------------------------------------------------

class ExponentialHealthReward(RewardFn):
    """
    Damage penalty grows as an exponential function of health lost.

    x = 1.0 - (health / max_health)   # 0 at full health, 1 at 0hp
    multiplier = exp(k * x)            # 1.0 at full health, e^k near death

    This produces a smooth curve that rises slowly at high health and
    accelerates steeply as health depletes — no discontinuous cliff,
    but strong urgency near death.

    k (steepness) controls where the curve "bends":
      k=2: gentle curve, moderate urgency at low health
      k=3: curve bends around 25hp (x=0.75) — recommended
      k=4: steep curve, very strong urgency below 25hp

    Penalty per 5hp damage hit (scale=0.5, k=3):
      100hp (x=0.00): 0.5 × exp(0.00) = 0.50
       75hp (x=0.25): 0.5 × exp(0.75) = 1.06
       50hp (x=0.50): 0.5 × exp(1.50) = 2.24
       25hp (x=0.75): 0.5 × exp(2.25) = 4.74
       10hp (x=0.90): 0.5 × exp(2.70) = 7.45
    """

    def __init__(
        self,
        death_penalty: float = -100.0,
        pipe_bonus:    float =   10.0,
        damage_scale:  float =    0.5,
        steepness:     float =    3.0,   # k — higher = steeper curve
        max_health:    float =  100.0,
    ):
        self.death_penalty = death_penalty
        self.pipe_bonus    = pipe_bonus
        self.damage_scale  = damage_scale
        self.steepness     = steepness
        self.max_health    = max_health

    def __call__(self, prev: GameState, curr: GameState, action: int) -> float:
        if curr.terminated:
            return self.death_penalty

        reward = 1.0

        # Pipe passing bonus
        if curr.score > prev.score:
            reward += self.pipe_bonus * (curr.score - prev.score)

        # Damage penalty — exponential urgency scaling
        damage_taken = prev.health - curr.health
        if damage_taken > 0:
            import math
            x          = 1.0 - (prev.health / self.max_health)
            multiplier = math.exp(self.steepness * x)
            reward    -= damage_taken * self.damage_scale * multiplier

        return reward