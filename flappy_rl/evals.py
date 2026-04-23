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
from algorithms import PPO, DQN, A2C

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
    p.add_argument("--config",        default=None,
                   help="Load scenario from a benchmark YAML config (e.g. benchmark_configs/stress_foam.yaml). "
                        "Sets exp, health, foam_pct, and gap overrides. --variant still required.")
    p.add_argument("--exp",           default=None,
                   help="Experiment name (e.g. baseline, foam, full)")
    p.add_argument("--variant",       required=True,
                   help="Variant name (e.g. ppo, ppo_v2, dqn_scored)")
    p.add_argument("--episodes",      type=int, default=3)
    p.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--health",        type=float, default=None,
                   help="Override starting health (e.g. 20.0) to test low-health behavior")
    p.add_argument("--foam-pct",      type=float, default=None,
                   help="Override foam pipe percentage 0.0-1.0 (e.g. 0.7). "
                        "Remaining weight split equally across hard/soft/brittle.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Resolve scenario from config file or inline args ---
    exp_name = args.exp
    health   = args.health
    foam_pct = args.foam_pct
    gap_overrides: dict = {}

    if args.config is not None:
        import random
        raw = yaml.safe_load(Path(args.config).read_text())
        if exp_name is None:
            exp_name = raw.get("exp")
        if health is None and "health" in raw:
            health = float(raw["health"])
        if foam_pct is None and "foam_pct" in raw:
            foam_pct = float(raw["foam_pct"])
        # Sample gap sizes once (same logic as benchmark, fixed seed for reproducibility)
        rng = random.Random(raw.get("seed", None))
        gap_raw = raw.get("gap", {}) or {}
        for key, opt_key in [("hard", "gap_hard"), ("foam", "gap_foam"),
                              ("soft", "gap_soft"), ("brittle", "gap_brittle")]:
            gcfg = gap_raw.get(key)
            if gcfg:
                if "value" in gcfg:
                    gap_overrides[opt_key] = float(gcfg["value"])
                else:
                    lo = float(gcfg.get("min", 0))
                    hi = float(gcfg.get("max", lo))
                    gap_overrides[opt_key] = lo if rng.random() < 0.5 else lo + rng.random() * (hi - lo)
        print(f"[eval] config     : {Path(args.config).name}")

    if exp_name is None:
        print("ERROR: --exp is required (or provide --config with an exp field).")
        return

    # --- Load exact env from training snapshot ---
    cfg      = _load_env(exp_name)
    run_info = _load_run_info(exp_name, args.variant)
    ckpt     = _best_checkpoint(exp_name, args.variant)

    # --- Pipe weight override ---
    if foam_pct is not None:
        foam = max(0.0, min(1.0, foam_pct))
        rest = (1.0 - foam) / 3.0
        cfg.pipe_weight_foam    = foam
        cfg.pipe_weight_hard    = rest
        cfg.pipe_weight_soft    = rest
        cfg.pipe_weight_brittle = rest

    obs_name    = run_info.get("obs_builder")
    obs_builder = _make_obs_builder(obs_name, cfg)
    algo        = _infer_algo(run_info, args.variant)

    ckpt_label = "best_model" if "best_model" in ckpt else Path(ckpt).name

    print(f"[eval] experiment : {exp_name}")
    print(f"[eval] variant    : {args.variant}")
    print(f"[eval] algo       : {algo.upper()}")
    print(f"[eval] checkpoint : {ckpt_label}")
    print(f"[eval] obs builder: {obs_builder.__class__.__name__}")
    print(f"[eval] episodes   : {args.episodes}")
    if health is not None:
        print(f"[eval] start health: {health} hp  (override)")
    if foam_pct is not None:
        print(f"[eval] pipe mix   : foam={cfg.pipe_weight_foam:.0%}  "
              f"hard={cfg.pipe_weight_hard:.0%}  soft={cfg.pipe_weight_soft:.0%}  "
              f"brittle={cfg.pipe_weight_brittle:.0%}  (override)")
    if gap_overrides:
        print(f"[eval] gaps       : {gap_overrides}")
    print("[eval] ESC/Q to quit early")

    model = ALGO_CLASSES[algo].load(ckpt)
    env   = FlappyBirdEnv(cfg, obs_builder=obs_builder, render_mode="human")
    clock = pygame.time.Clock()
    ep_rewards = []

    for ep in range(args.episodes):
        options: dict = {}
        if health is not None:
            options["health"] = health
        options.update(gap_overrides)
        obs, _ = env.reset(options=options if options else None)
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
