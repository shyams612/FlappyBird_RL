# train.py
"""
Unified training entry point for PPO, DQN, and A2C.

New experiment structure:
    runs/<exp_name>/                  ← shared across all algos
        env_config.yaml               ← full env snapshot
        experiment.yaml               ← env description, features
        ppo/                          ← one folder per algo run
            best_model.zip
            final_model.zip
            monitor.csv
            checkpoints/
            tensorboard/
            run_info.yaml             ← algo, hyperparams, timestamp
        dqn/
        a2c/

Usage:
    python train.py --exp baseline --algo ppo
    python train.py --exp foam --algo dqn
    python train.py --exp full --algo a2c --timesteps 2000000
"""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import yaml

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.flappy_env import FlappyBirdEnv
from envs.config import EnvConfig
from envs.observations import SimpleObsBuilder, Config2ObsBuilder, Config2NoisyObsBuilder
from envs.rewards import SurvivalReward, HealthAwareReward


# ---------------------------------------------------------------------------
# Per-algorithm hyperparameters
# ---------------------------------------------------------------------------

ALGO_PARAMS = {
    "ppo": dict(
        policy        = "MlpPolicy",
        learning_rate = 3e-4,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        verbose       = 1,
    ),
    "dqn": dict(
        policy                 = "MlpPolicy",
        learning_rate          = 1e-4,
        buffer_size            = 100_000,
        learning_starts        = 10_000,
        batch_size             = 64,
        tau                    = 1.0,
        gamma                  = 0.99,
        train_freq             = 4,
        gradient_steps         = 1,
        target_update_interval = 1000,
        exploration_fraction   = 0.1,
        exploration_final_eps  = 0.05,
        verbose                = 1,
    ),
    "a2c": dict(
        policy        = "MlpPolicy",
        learning_rate = 7e-4,
        n_steps       = 128,       # increased from default 5 — better credit assignment
        gamma         = 0.99,
        gae_lambda    = 1.0,
        ent_coef      = 0.0,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        verbose       = 1,
    ),
}

ALGO_CLASSES = {"ppo": PPO, "dqn": DQN, "a2c": A2C}

# Maps experiment name → config file
EXP_CONFIGS = {
    "baseline":       "config/env_baseline.yaml",
    "foam":           "config/env_foam.yaml",
    "foam_kits":      "config/env_foam_kits.yaml",
    "foam_kits_wind": "config/env_foam_kits_wind.yaml",
    "full":           "config/env_full.yaml",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_obs_builder(cfg: EnvConfig, obs_override: str | None = None):
    if obs_override == "config2_noisy":
        return Config2NoisyObsBuilder(cfg)
    if obs_override == "config2":
        return Config2ObsBuilder(cfg)
    if obs_override == "simple":
        return SimpleObsBuilder(cfg)
    if cfg.enable_pipe_variants:
        return Config2ObsBuilder(cfg)
    return SimpleObsBuilder(cfg)


def make_reward_fn(cfg: EnvConfig):
    if cfg.enable_pipe_variants:
        return HealthAwareReward()
    return SurvivalReward()


def _env_feature_summary(cfg: EnvConfig) -> dict:
    return {
        "pipe_variants":    cfg.enable_pipe_variants,
        "gradient_damage":  cfg.enable_gradient_damage,
        "bullets":          cfg.enable_bullets,
        "wind":             cfg.enable_wind,
        "health_kits":      cfg.enable_health_kits,
        "passive_drain":    cfg.passive_drain,
        "pipe_weights": {
            "hard":    cfg.pipe_weight_hard,
            "soft":    cfg.pipe_weight_soft,
            "brittle": cfg.pipe_weight_brittle,
            "foam":    cfg.pipe_weight_foam,
        },
        "gap_heights": {
            "hard":    cfg.gap_height,
            "soft":    cfg.gap_height_soft,
            "brittle": cfg.gap_height_brittle,
            "foam":    cfg.gap_height_foam,
        },
        "scroll_speed":  cfg.scroll_speed,
        "health_start":  cfg.health_start,
        "bullet_count":  cfg.bullet_count,
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp",       default=None,
                   help=f"Experiment name: {list(EXP_CONFIGS.keys())}")
    p.add_argument("--algo",      choices=["ppo", "dqn", "a2c"], default="ppo")
    p.add_argument("--config",    default=None,
                   help="Path to YAML env config (overrides --exp lookup)")
    p.add_argument("--obs",       choices=["simple", "config2", "config2_noisy"],
                   default=None)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--save-freq", type=int, default=100_000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Resolve config ---
    if args.config:
        config_path = args.config
        exp_name    = Path(args.config).stem  # use filename as exp name
    elif args.exp:
        if args.exp not in EXP_CONFIGS:
            print(f"[train] Unknown experiment '{args.exp}'. "
                  f"Choose from: {list(EXP_CONFIGS.keys())}")
            return
        config_path = EXP_CONFIGS[args.exp]
        exp_name    = args.exp
    else:
        print("[train] Provide --exp <name> or --config <path>")
        return

    cfg = EnvConfig.from_yaml(config_path)

    # --- Directory structure: runs/<exp_name>/<algo>/ ---
    exp_dir  = Path("runs") / exp_name
    algo_dir = exp_dir / args.algo
    algo_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] experiment  : {exp_name}")
    print(f"[train] algo        : {args.algo.upper()}")
    print(f"[train] run dir     : {algo_dir}")
    print(f"[train] timesteps   : {args.timesteps:,}")

    # --- Obs / reward ---
    obs_builder = make_obs_builder(cfg, args.obs)
    reward_fn   = make_reward_fn(cfg)
    print(f"[train] obs builder : {obs_builder.__class__.__name__}")
    print(f"[train] reward fn   : {reward_fn.__class__.__name__}")

    # --- Shared experiment metadata (written once, not overwritten if exists) ---
    exp_meta_path = exp_dir / "experiment.yaml"
    if not exp_meta_path.exists():
        cfg.to_yaml(exp_dir / "env_config.yaml")
        exp_meta = {
            "exp_name":    exp_name,
            "config_file": config_path,
            "features":    _env_feature_summary(cfg),
        }
        with open(exp_meta_path, "w") as f:
            yaml.dump(exp_meta, f, sort_keys=False)
        print(f"[train] env config  : {exp_dir}/env_config.yaml")

    # --- Per-algo run info ---
    run_info = {
        "algo":        args.algo.upper(),
        "timestamp":   datetime.now().isoformat(),
        "timesteps":   args.timesteps,
        "seed":        args.seed,
        "obs_builder": obs_builder.__class__.__name__,
        "reward_fn":   reward_fn.__class__.__name__,
        "hyperparams": ALGO_PARAMS[args.algo],
    }
    with open(algo_dir / "run_info.yaml", "w") as f:
        yaml.dump(run_info, f, sort_keys=False)

    # --- Environments ---
    train_env = Monitor(
        FlappyBirdEnv(cfg, obs_builder=make_obs_builder(cfg, args.obs),
                      reward_fn=make_reward_fn(cfg)),
        filename=str(algo_dir / "monitor.csv"),
    )
    eval_env = Monitor(
        FlappyBirdEnv(cfg, obs_builder=make_obs_builder(cfg, args.obs),
                      reward_fn=make_reward_fn(cfg)),
    )

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = str(algo_dir),
        log_path             = str(algo_dir / "eval"),
        eval_freq            = args.eval_freq,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq   = args.save_freq,
        save_path   = str(algo_dir / "checkpoints"),
        name_prefix = args.algo,
        verbose     = 1,
    )

    # --- Model ---
    AlgoClass = ALGO_CLASSES[args.algo]
    params    = ALGO_PARAMS[args.algo].copy()
    params.update(
        env             = train_env,
        seed            = args.seed,
        tensorboard_log = str(algo_dir / "tensorboard"),
    )
    model = AlgoClass(**params)

    print(f"[train] obs space   : {train_env.observation_space}")
    print(f"[train] action space: {train_env.action_space}")

    # --- Train ---
    model.learn(
        total_timesteps = args.timesteps,
        callback        = [eval_callback, checkpoint_callback],
        progress_bar    = True,
    )

    # --- Save ---
    final_path = str(algo_dir / "final_model")
    model.save(final_path)
    print(f"[train] final model saved → {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
