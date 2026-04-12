# evals.py
"""
Load a saved model and run the agent visually.

Automatically reconstructs the exact environment used during training
from the experiment's env_config.yaml and variant's run_info.yaml.

Usage:
    python evals.py --exp baseline --variant ppo
    python evals.py --exp foam --variant ppo_v2 --episodes 5
    python evals.py --exp baseline --variant dqn --deterministic false
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import yaml
import pygame
from stable_baselines3 import PPO, DQN, A2C

from envs.flappy_env import FlappyBirdEnv
from envs.config import EnvConfig
from envs.observations import SimpleObsBuilder, Config2ObsBuilder, Config2NoisyObsBuilder

ALGO_CLASSES = {"ppo": PPO, "dqn": DQN, "a2c": A2C}
RUNS_DIR     = Path("runs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env(exp_name: str) -> EnvConfig:
    """Load env_config.yaml from the experiment directory."""
    path = RUNS_DIR / exp_name / "env_config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"env_config.yaml not found for experiment '{exp_name}'. "
            f"Has it been trained yet?"
        )
    return EnvConfig.from_yaml(path)


def _load_run_info(exp_name: str, variant: str) -> dict:
    path = RUNS_DIR / exp_name / variant / "run_info.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _best_checkpoint(exp_name: str, variant: str) -> str:
    variant_dir = RUNS_DIR / exp_name / variant
    best = variant_dir / "best_model.zip"
    if best.exists():
        return str(variant_dir / "best_model")
    ckpt_dir = variant_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return str(ckpts[-1].with_suffix(""))
    raise FileNotFoundError(
        f"No checkpoint found for {exp_name}/{variant}. Train it first."
    )


def _make_obs_builder(name: str, cfg: EnvConfig):
    if name == "Config2NoisyObsBuilder":
        return Config2NoisyObsBuilder(cfg)
    if name == "Config2ObsBuilder":
        return Config2ObsBuilder(cfg)
    # Default / SimpleObsBuilder
    if cfg.enable_pipe_variants:
        return Config2ObsBuilder(cfg)
    return SimpleObsBuilder(cfg)


def _infer_algo(run_info: dict, variant: str) -> str:
    """Get algo from run_info, or guess from variant name prefix."""
    algo = run_info.get("algo", "").lower()
    if algo in ALGO_CLASSES:
        return algo
    for name in ALGO_CLASSES:
        if variant.startswith(name):
            return name
    raise ValueError(
        f"Cannot determine algo for variant '{variant}'. "
        f"run_info.yaml missing or variant name doesn't start with algo name."
    )


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp",           required=True,
                   help="Experiment name (e.g. baseline, foam, full)")
    p.add_argument("--variant",       required=True,
                   help="Variant name (e.g. ppo, ppo_v2, dqn_scored)")
    p.add_argument("--episodes",      type=int, default=3)
    p.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Load exact env from training snapshot ---
    cfg      = _load_env(args.exp)
    run_info = _load_run_info(args.exp, args.variant)
    ckpt     = _best_checkpoint(args.exp, args.variant)

    obs_name    = run_info.get("obs_builder")
    obs_builder = _make_obs_builder(obs_name, cfg)
    algo        = _infer_algo(run_info, args.variant)

    ckpt_label = "best_model" if "best_model" in ckpt else Path(ckpt).name

    print(f"[eval] experiment : {args.exp}")
    print(f"[eval] variant    : {args.variant}")
    print(f"[eval] algo       : {algo.upper()}")
    print(f"[eval] checkpoint : {ckpt_label}")
    print(f"[eval] obs builder: {obs_builder.__class__.__name__}")
    print(f"[eval] episodes   : {args.episodes}")
    print("[eval] ESC/Q to quit early")

    model = ALGO_CLASSES[algo].load(ckpt)
    env   = FlappyBirdEnv(cfg, obs_builder=obs_builder, render_mode="human")
    clock = pygame.time.Clock()
    ep_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated

            env.render()
            clock.tick(cfg.fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close(); return
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        env.close(); return

        ep_rewards.append(total_reward)
        print(f"[eval] episode {ep+1}: reward={total_reward:.1f}  "
              f"score={info['score']}  frames={info['frame']}")
        time.sleep(1.0)

    print(f"\n[eval] mean reward : {sum(ep_rewards)/len(ep_rewards):.1f}")
    env.close()


if __name__ == "__main__":
    main()
