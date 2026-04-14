# benchmark.py
"""
Headless multi-episode benchmark for comparing trained variants.

Runs N episodes per variant without rendering and reports:
  mean/std reward, mean/std episode length, mean/std pipes cleared,
  survival rate (% episodes not dying in first 100 frames).

Usage:
    # Single variant
    python benchmark.py --exp foam --variants ppo_exponential --episodes 100

    # Compare multiple variants
    python benchmark.py --exp foam --variants ppo2 ppo_threshold ppo_exponential --episodes 100

    # Stress-test health-aware behavior
    python benchmark.py --exp foam --variants ppo2 ppo_threshold ppo_exponential \
        --episodes 100 --health 20 --foam-pct 0.7

    # Config-file-driven benchmark
    python benchmark.py --config benchmark_configs/stress_foam.yaml
"""

from __future__ import annotations
import argparse
import math
import random
from pathlib import Path

import yaml
from stable_baselines3 import PPO, DQN, A2C

from envs.flappy_env import FlappyBirdEnv
from envs.config import EnvConfig
from envs.observations import SimpleObsBuilder, Config2ObsBuilder, Config2NoisyObsBuilder

ALGO_CLASSES = {"ppo": PPO, "dqn": DQN, "a2c": A2C}
RUNS_DIR     = Path("runs")


# ---------------------------------------------------------------------------
# Helpers (shared with evals.py)
# ---------------------------------------------------------------------------

def _load_env(exp_name: str) -> EnvConfig:
    path = RUNS_DIR / exp_name / "env_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"env_config.yaml not found for '{exp_name}'.")
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
    raise FileNotFoundError(f"No checkpoint found for {exp_name}/{variant}.")


def _make_obs_builder(name: str, cfg: EnvConfig):
    if name == "Config2NoisyObsBuilder":
        return Config2NoisyObsBuilder(cfg)
    if name == "Config2ObsBuilder":
        return Config2ObsBuilder(cfg)
    if cfg.enable_pipe_variants:
        return Config2ObsBuilder(cfg)
    return SimpleObsBuilder(cfg)


def _infer_algo(run_info: dict, variant: str) -> str:
    algo = run_info.get("algo", "").lower()
    if algo in ALGO_CLASSES:
        return algo
    for name in ALGO_CLASSES:
        if variant.startswith(name):
            return name
    raise ValueError(f"Cannot determine algo for variant '{variant}'.")


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def _sample_gap(gap_cfg: dict | None, default: float, rng: random.Random) -> float | None:
    """
    Sample a gap size for one episode from a gap config block.

    gap_cfg format:
        value: 140          # fixed — same every episode
      OR
        min: 100            # 50% of episodes use min, 50% use uniform(min, max)
        max: 160

    Returns None if gap_cfg is None (use env default).
    """
    if gap_cfg is None:
        return None
    if "value" in gap_cfg:
        return float(gap_cfg["value"])
    lo = float(gap_cfg.get("min", default))
    hi = float(gap_cfg.get("max", default))
    if rng.random() < 0.5:
        return lo
    return lo + rng.random() * (hi - lo)


def run_benchmark(
    exp_name: str,
    variant: str,
    n_episodes: int = 100,
    health_override: float | None = None,
    foam_pct: float | None = None,
    max_frames: int | None = None,
    gap_hard: dict | None = None,
    gap_foam: dict | None = None,
    gap_soft: dict | None = None,
    gap_brittle: dict | None = None,
    deterministic: bool = True,
    seed: int | None = None,
) -> dict:
    """
    Run n_episodes headlessly and return stats dict.

    gap_hard/foam/soft/brittle accept dicts with either:
      {"value": N}       — fixed gap every episode
      {"min": N, "max": M} — 50% at min, 50% uniform(min, max)
    """
    cfg      = _load_env(exp_name)
    run_info = _load_run_info(exp_name, variant)
    ckpt     = _best_checkpoint(exp_name, variant)

    # Pipe weight override
    if foam_pct is not None:
        foam = max(0.0, min(1.0, foam_pct))
        rest = (1.0 - foam) / 3.0
        cfg.pipe_weight_foam    = foam
        cfg.pipe_weight_hard    = rest
        cfg.pipe_weight_soft    = rest
        cfg.pipe_weight_brittle = rest

    obs_name    = run_info.get("obs_builder")
    obs_builder = _make_obs_builder(obs_name, cfg)
    algo        = _infer_algo(run_info, variant)
    reward_fn_name = run_info.get("reward_fn", "?")

    model = ALGO_CLASSES[algo].load(ckpt)
    env   = FlappyBirdEnv(cfg, obs_builder=obs_builder)  # headless

    rng = random.Random(seed)
    rewards, ep_lengths, scores, min_healths = [], [], [], []

    for _ in range(n_episodes):
        options: dict = {}
        if health_override is not None:
            options["health"] = health_override
        if max_frames is not None:
            options["max_frames"] = max_frames

        # Per-episode gap sampling
        g_hard    = _sample_gap(gap_hard,    cfg.gap_height,         rng)
        g_foam    = _sample_gap(gap_foam,    cfg.gap_height_foam,    rng)
        g_soft    = _sample_gap(gap_soft,    cfg.gap_height_soft,    rng)
        g_brittle = _sample_gap(gap_brittle, cfg.gap_height_brittle, rng)
        if g_hard    is not None: options["gap_hard"]    = g_hard
        if g_foam    is not None: options["gap_foam"]    = g_foam
        if g_soft    is not None: options["gap_soft"]    = g_soft
        if g_brittle is not None: options["gap_brittle"] = g_brittle

        obs, _ = env.reset(options=options if options else None)
        total_reward = 0.0
        done         = False
        min_health   = health_override if health_override is not None else cfg.health_start

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done          = terminated or truncated
            min_health    = min(min_health, info.get("health", min_health))

        rewards.append(total_reward)
        ep_lengths.append(info["frame"])
        scores.append(info["score"])
        min_healths.append(min_health)

    env.close()

    mean_r   = sum(rewards)     / n_episodes
    mean_len = sum(ep_lengths)  / n_episodes
    mean_sc  = sum(scores)      / n_episodes
    mean_mh  = sum(min_healths) / n_episodes
    # Survival rate: episodes where agent lasted more than 200 frames
    survived = sum(1 for l in ep_lengths if l > 200) / n_episodes

    return {
        "variant":      variant,
        "algo":         algo,
        "reward_fn":    reward_fn_name,
        "episodes":     n_episodes,
        "mean_reward":  mean_r,
        "std_reward":   _std(rewards),
        "mean_ep_len":  mean_len,
        "std_ep_len":   _std(ep_lengths),
        "mean_score":   mean_sc,
        "std_score":    _std(scores),
        "mean_min_health": mean_mh,
        "survival_rate": survived,
    }


# ---------------------------------------------------------------------------
# Config-file-driven benchmark
# ---------------------------------------------------------------------------

def run_benchmark_config(config_path: str | Path) -> list[dict]:
    """
    Load a YAML benchmark config and run all variants.

    Config format:
        exp: foam
        episodes: 50
        deterministic: true
        seed: 42                    # optional — for reproducible gap sampling
        max_frames: 3000            # optional — cap per episode
        health: 20.0                # optional — starting health override
        foam_pct: 0.7               # optional — pipe weight override
        gap:                        # optional — gap size config per pipe type
          hard:
            min: 120
            max: 160
          foam:
            value: 110
          soft:
            min: 80
            max: 110
          brittle:
            min: 70
            max: 100
        variants:
          - ppo2
          - ppo_threshold
          - ppo_exponential
    """
    raw = yaml.safe_load(Path(config_path).read_text())

    exp_name      = raw["exp"]
    n_episodes    = int(raw.get("episodes", 100))
    deterministic = bool(raw.get("deterministic", True))
    seed          = raw.get("seed", None)
    health        = raw.get("health", None)
    foam_pct      = raw.get("foam_pct", None)
    max_frames    = raw.get("max_frames", None)
    variants      = raw.get("variants", [])

    gap_raw     = raw.get("gap", {}) or {}
    gap_hard    = gap_raw.get("hard",    None)
    gap_foam    = gap_raw.get("foam",    None)
    gap_soft    = gap_raw.get("soft",    None)
    gap_brittle = gap_raw.get("brittle", None)

    results = []
    for variant in variants:
        print(f"  Running {variant}...", end="", flush=True)
        try:
            r = run_benchmark(
                exp_name      = exp_name,
                variant       = variant,
                n_episodes    = n_episodes,
                health_override = float(health) if health is not None else None,
                foam_pct      = float(foam_pct) if foam_pct is not None else None,
                max_frames    = int(max_frames) if max_frames is not None else None,
                gap_hard      = gap_hard,
                gap_foam      = gap_foam,
                gap_soft      = gap_soft,
                gap_brittle   = gap_brittle,
                deterministic = deterministic,
                seed          = seed,
            )
            results.append(r)
            print(f" done  (mean reward {r['mean_reward']:.0f})")
        except Exception as e:
            print(f" ERROR: {e}")
    return results


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Config-file mode (mutually exclusive with inline args)
    p.add_argument("--config",        default=None,
                   help="Path to a YAML benchmark config file")
    # Inline mode
    p.add_argument("--exp",           default=None)
    p.add_argument("--variants",      nargs="+", default=None,
                   help="One or more variant names to benchmark")
    p.add_argument("--episodes",      type=int,   default=100)
    p.add_argument("--health",        type=float, default=None,
                   help="Override starting health for all episodes")
    p.add_argument("--foam-pct",      type=float, default=None,
                   help="Override foam pipe fraction 0.0–1.0")
    p.add_argument("--max-frames",    type=int,   default=None,
                   help="Cap episode length (frames) to speed up benchmarks")
    p.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--seed",          type=int,   default=None,
                   help="RNG seed for reproducible gap sampling")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_results(results: list[dict]) -> None:
    if not results:
        return
    col = 18
    header = (f"  {'VARIANT':<{col}} {'REWARD':>10} {'±':>6} {'EP LEN':>8} "
              f"{'±':>6} {'SCORE':>7} {'MIN HP':>7} {'SURVIVE':>8}  REWARD_FN")
    sep    = "  " + "─" * (len(header) - 2)
    print(f"\n{header}")
    print(sep)
    for r in results:
        print(
            f"  {r['variant']:<{col}} "
            f"{r['mean_reward']:>10.1f} "
            f"{r['std_reward']:>6.0f} "
            f"{r['mean_ep_len']:>8.0f} "
            f"{r['std_ep_len']:>6.0f} "
            f"{r['mean_score']:>7.1f} "
            f"{r['mean_min_health']:>7.1f} "
            f"{r['survival_rate']:>7.0%}  "
            f"{r['reward_fn']}"
        )
    print()


def main() -> None:
    args = parse_args()

    # --- Config-file mode ---
    if args.config is not None:
        cfg_path = Path(args.config)
        raw = yaml.safe_load(cfg_path.read_text())
        print(f"\nBenchmark config: {cfg_path.name}  |  exp={raw.get('exp')}  "
              f"|  {raw.get('episodes', 100)} episodes per variant\n")
        results = run_benchmark_config(cfg_path)
        _print_results(results)
        return

    # --- Inline mode ---
    if not args.exp or not args.variants:
        print("ERROR: provide either --config FILE or both --exp and --variants.")
        return

    print(f"\nBenchmark: {args.exp}  |  {args.episodes} episodes per variant")
    if args.health is not None:
        print(f"  start health : {args.health} hp")
    if args.foam_pct is not None:
        print(f"  foam pct     : {args.foam_pct:.0%}")
    if args.max_frames is not None:
        print(f"  max frames   : {args.max_frames}")
    print()

    results = []
    for variant in args.variants:
        print(f"  Running {variant}...", end="", flush=True)
        try:
            r = run_benchmark(
                exp_name        = args.exp,
                variant         = variant,
                n_episodes      = args.episodes,
                health_override = args.health,
                foam_pct        = args.foam_pct,
                max_frames      = args.max_frames,
                deterministic   = args.deterministic,
                seed            = args.seed,
            )
            results.append(r)
            print(f" done  (mean reward {r['mean_reward']:.0f})")
        except Exception as e:
            print(f" ERROR: {e}")

    _print_results(results)


if __name__ == "__main__":
    main()
