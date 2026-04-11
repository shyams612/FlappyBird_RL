# eval.py
"""
Load a saved checkpoint and run the agent visually.

Usage:
    python eval.py                                              # auto-detect latest run + algo
    python eval.py --algo dqn                                  # latest DQN run
    python eval.py --checkpoint runs/ppo_20240330/best_model   # explicit path (algo auto-detected)
    python eval.py --checkpoint runs/dqn_20240330/best_model --algo dqn
    python eval.py --episodes 5 --deterministic false
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import pygame
from stable_baselines3 import PPO, DQN, A2C

from envs.flappy_env import FlappyBirdEnv
from envs.config import EnvConfig
from envs.observations import SimpleObsBuilder, Config2ObsBuilder, Config2NoisyObsBuilder

ALGO_CLASSES = {"ppo": PPO, "dqn": DQN, "a2c": A2C}


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def latest_checkpoint(runs_dir: str = "runs", algo: str | None = None) -> tuple[str, str]:
    """
    Returns (checkpoint_path, algo_name) for the most recently modified
    run folder, optionally filtered by algo prefix.

    Resolution order per run folder:
      1. best_model.zip
      2. latest step checkpoint in checkpoints/
    """
    runs_dir_path = Path(runs_dir)
    if not runs_dir_path.exists():
        raise FileNotFoundError(f"No runs directory found at '{runs_dir}'")

    runs = sorted(runs_dir_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = [r for r in runs if r.is_dir()]

    if algo:
        runs = [r for r in runs if r.name.startswith(algo)]

    if not runs:
        suffix = f" for algo '{algo}'" if algo else ""
        raise FileNotFoundError(f"No run folders found{suffix} in '{runs_dir}'")

    for run in runs:
        detected_algo = _detect_algo(run)

        best = run / "best_model.zip"
        if best.exists():
            return str(run / "best_model"), detected_algo

        ckpt_dir = run / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
            if ckpts:
                return str(ckpts[-1].with_suffix("")), detected_algo

    raise FileNotFoundError(f"No checkpoints found in '{runs_dir}'")


def _detect_algo(run_dir: Path) -> str:
    """Infer algorithm from the run folder name prefix (e.g. 'ppo_20240330' → 'ppo')."""
    for name in ALGO_CLASSES:
        if run_dir.name.startswith(name):
            return name
    return "ppo"   # safe default


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default=None,
                   help="Path to model .zip (omit extension). Defaults to latest.")
    p.add_argument("--algo",          default=None, choices=list(ALGO_CLASSES),
                   help="ppo / dqn / a2c. Auto-detected from run folder name if omitted.")
    p.add_argument("--config",        default=None, help="Path to YAML env config")
    p.add_argument("--obs",           choices=["simple", "config2", "config2_noisy"], default=None,
                   help="Override obs builder (default: auto-selected from config)")
    p.add_argument("--episodes",      type=int, default=3)
    p.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=True,
                   help="Use deterministic actions (default: true)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Resolve checkpoint + algo ---
    if args.checkpoint:
        ckpt  = args.checkpoint
        algo  = args.algo or _detect_algo(Path(ckpt).parent)
    else:
        ckpt, algo = latest_checkpoint(algo=args.algo)

    if args.algo and args.algo != algo:
        algo = args.algo   # explicit flag always wins

    # --- Load ---
    cfg = EnvConfig.from_yaml(args.config) if args.config else EnvConfig()

    # Obs builder — explicit flag takes precedence over auto-selection
    obs_override = getattr(args, 'obs', None)
    if obs_override == "config2_noisy":
        obs_builder = Config2NoisyObsBuilder(cfg)
    elif obs_override == "config2":
        obs_builder = Config2ObsBuilder(cfg)
    elif obs_override == "simple":
        obs_builder = SimpleObsBuilder(cfg)
    elif cfg.enable_pipe_variants:
        obs_builder = Config2ObsBuilder(cfg)
    else:
        obs_builder = SimpleObsBuilder(cfg)

    model = ALGO_CLASSES[algo].load(ckpt)
    env   = FlappyBirdEnv(cfg, obs_builder=obs_builder, render_mode="human")
    print(f"[eval] obs builder : {obs_builder.__class__.__name__}")

    print(f"[eval] algo       : {algo.upper()}")
    print(f"[eval] checkpoint : {ckpt}")
    print(f"[eval] episodes   : {args.episodes}")
    print(f"[eval] deterministic: {args.deterministic}")
    print("[eval] ESC/Q to quit early")

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
        print(f"[eval] episode {ep+1}: reward={total_reward:.1f}  score={info['score']}  frames={info['frame']}")
        time.sleep(1.0)

    print(f"\n[eval] mean reward over {args.episodes} episodes: {sum(ep_rewards)/len(ep_rewards):.1f}")
    env.close()


if __name__ == "__main__":
    main()