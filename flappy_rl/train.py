# train.py
"""
Unified training entry point for PPO, DQN, and A2C.

Experiment structure:
    runs/<exp_name>/
        env_config.yaml               ← full env snapshot (shared, immutable)
        experiment.yaml               ← env features summary (shared)
        patches/                      ← hyperparameter patch files
            ppo_v2.yaml               ← {ent_coef: 0.01, learning_rate: 1e-4}
        <variant>/                    ← e.g. ppo, ppo_v2, dqn, dqn_scored
            best_model.zip
            final_model.zip
            monitor.csv
            checkpoints/
            tensorboard/
            run_info.yaml             ← algo, variant, patch_diff, duration, etc.

Usage:
    python train.py --exp baseline --algo ppo
    python train.py --exp baseline --algo ppo --variant ppo_v2
    python train.py --exp foam --algo dqn --variant dqn_scored
"""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import yaml

from algorithms import PPO, DQN, A2C
from algorithms.monitor import Monitor
from algorithms.callbacks import EvalCallback, CheckpointCallback

from envs.flappy_env import FlappyBirdEnv
from envs.config import EnvConfig
from envs.observations import SimpleObsBuilder, Config2ObsBuilder, Config2NoisyObsBuilder
from envs.rewards import SurvivalReward, HealthAwareReward, ScoredReward, ThresholdHealthReward, ExponentialHealthReward, AsymmetricExponentialReward


# ---------------------------------------------------------------------------
# Base hyperparameters
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
        ent_coef      = 0.0,
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
        n_steps       = 128,
        gamma         = 0.99,
        gae_lambda    = 1.0,
        ent_coef      = 0.0,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        verbose       = 1,
    ),
}

ALGO_CLASSES = {"ppo": PPO, "dqn": DQN, "a2c": A2C}

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


def make_reward_fn(cfg: EnvConfig, reward_override: str | None = None):
    if reward_override == "scored":
        return ScoredReward()
    if reward_override == "survival":
        return SurvivalReward()
    if reward_override == "health_aware" or reward_override == "continuous":
        return HealthAwareReward(damage_scale=cfg.health_reward_scale)
    if reward_override == "threshold_health" or reward_override == "threshold":
        return ThresholdHealthReward(
            damage_scale=cfg.health_reward_scale,
            threshold=cfg.health_reward_threshold,
            threshold_scale=cfg.health_reward_threshold_scale,
        )
    if reward_override == "exponential_health" or reward_override == "exponential":
        return ExponentialHealthReward(
            damage_scale=cfg.health_reward_scale,
            steepness=cfg.health_reward_steepness,
        )
    if reward_override == "asymmetric" or reward_override == "asymmetric_exponential":
        return AsymmetricExponentialReward(
            scale=cfg.health_reward_scale,
            steepness=cfg.health_reward_steepness,
            crossover=cfg.health_reward_crossover,
        )
    # Fall back to config default
    if cfg.health_reward_fn == "continuous":
        return HealthAwareReward(damage_scale=cfg.health_reward_scale)
    if cfg.health_reward_fn == "threshold":
        return ThresholdHealthReward(
            damage_scale=cfg.health_reward_scale,
            threshold=cfg.health_reward_threshold,
            threshold_scale=cfg.health_reward_threshold_scale,
        )
    if cfg.health_reward_fn == "exponential":
        return ExponentialHealthReward(
            damage_scale=cfg.health_reward_scale,
            steepness=cfg.health_reward_steepness,
        )
    if cfg.health_reward_fn == "asymmetric":
        return AsymmetricExponentialReward(
            scale=cfg.health_reward_scale,
            steepness=cfg.health_reward_steepness,
            crossover=cfg.health_reward_crossover,
        )
    if cfg.enable_pipe_variants:
        return HealthAwareReward(damage_scale=cfg.health_reward_scale)
    return SurvivalReward()


def _env_feature_summary(cfg: EnvConfig) -> dict:
    return {
        "pipe_variants":   cfg.enable_pipe_variants,
        "gradient_damage": cfg.enable_gradient_damage,
        "bullets":         cfg.enable_bullets,
        "wind":            cfg.enable_wind,
        "health_kits":     cfg.enable_health_kits,
        "passive_drain":   cfg.passive_drain,
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


def _load_patch(patch_path: Path) -> dict:
    """Load a patch YAML and return the dict of overrides."""
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
    with open(patch_path) as f:
        return yaml.safe_load(f) or {}


def _apply_patch(base_params: dict, patch: dict) -> tuple[dict, dict]:
    """
    Apply patch overrides to base_params.
    Returns (merged_params, diff) where diff contains only changed keys.
    Special keys handled separately (not passed to SB3):
      reward_fn, obs_builder, timesteps
    """
    SB3_SKIP = {"reward_fn", "obs_builder", "timesteps"}
    merged = base_params.copy()
    diff   = {}
    for k, v in patch.items():
        if k in SB3_SKIP:
            continue
        if merged.get(k) != v:
            diff[k] = v
        merged[k] = v
    return merged, diff


def _latest_checkpoint(variant_dir: Path, algo: str) -> Path | None:
    ckpt_dir = variant_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob(f"{algo}_*_steps.zip"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def _completed_steps(variant_dir: Path, algo: str) -> int:
    """Infer how many steps were completed from the latest checkpoint filename."""
    ckpt = _latest_checkpoint(variant_dir, algo)
    if not ckpt:
        return 0
    try:
        return int(ckpt.stem.split("_")[1])
    except (IndexError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp",       default=None,
                   help=f"Experiment name: {list(EXP_CONFIGS.keys())}")
    p.add_argument("--algo",      choices=["ppo", "dqn", "a2c"], default="ppo")
    p.add_argument("--variant",   default=None,
                   help="Variant name (e.g. ppo_v2). Defaults to algo name.")
    p.add_argument("--patch",     default=None,
                   help="Path to patch YAML relative to exp dir "
                        "(e.g. patches/ppo_v2.yaml). Auto-resolved if variant matches patch name.")
    p.add_argument("--config",    default=None,
                   help="Path to YAML env config (overrides --exp lookup)")
    p.add_argument("--obs",       choices=["simple", "config2", "config2_noisy"], default=None)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--save-freq", type=int, default=100_000)
    p.add_argument("--resume",    action="store_true",
                   help="Resume from latest checkpoint without prompting")
    p.add_argument("--no-resume", action="store_true",
                   help="Skip resume prompt and start fresh")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Resolve config ---
    if args.config:
        config_path = args.config
        exp_name    = Path(args.config).stem
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

    # --- Variant name defaults to algo name ---
    variant = args.variant or args.algo

    # --- Directory structure: runs/<exp>/<variant>/ ---
    exp_dir     = Path("runs") / exp_name
    variant_dir = exp_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] experiment  : {exp_name}")
    print(f"[train] algo        : {args.algo.upper()}")
    print(f"[train] variant     : {variant}")
    print(f"[train] timesteps   : {args.timesteps:,}")

    # --- Load and apply patch ---
    patch_dict  = {}
    patch_diff  = {}
    patch_file  = None

    if args.patch:
        patch_file = exp_dir / args.patch
    else:
        # Auto-resolve: if variant != algo, look for patches/<variant>.yaml
        auto_patch = exp_dir / "patches" / f"{variant}.yaml"
        if variant != args.algo and auto_patch.exists():
            patch_file = auto_patch

    if patch_file:
        patch_dict = _load_patch(patch_file)
        print(f"[train] patch       : {patch_file}")
        for k, v in patch_dict.items():
            print(f"[train]   {k} = {v}")

    # --- Base hyperparams + patch ---
    base_params = ALGO_PARAMS[args.algo].copy()
    sb3_params, patch_diff = _apply_patch(base_params, patch_dict)

    # --- timesteps: patch overrides CLI arg ---
    if "timesteps" in patch_dict:
        args.timesteps = int(patch_dict["timesteps"])
        print(f"[train] timesteps   : {args.timesteps:,}  (overridden by patch)")

    # --- Reward / obs (patchable via patch yaml) ---
    reward_override = patch_dict.get("reward_fn")
    obs_override    = args.obs or patch_dict.get("obs_builder")
    obs_builder     = make_obs_builder(cfg, obs_override)
    reward_fn       = make_reward_fn(cfg, reward_override)
    print(f"[train] obs builder : {obs_builder.__class__.__name__}")
    print(f"[train] reward fn   : {reward_fn.__class__.__name__}")

    # --- Shared experiment metadata (written once) ---
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

    # --- Resume detection ---
    resume_from  = None
    done_steps   = _completed_steps(variant_dir, args.algo)
    is_partial   = done_steps > 0 and not (variant_dir / "final_model.zip").exists()

    if is_partial and not args.no_resume:
        ckpt = _latest_checkpoint(variant_dir, args.algo)
        if args.resume:
            resume_from = ckpt
        else:
            print(f"\n[train] Partial run detected — {done_steps:,} / {args.timesteps:,} steps completed")
            print(f"[train] Latest checkpoint: {ckpt.name}")
            ans = input("[train] Resume from checkpoint? [y/n]: ").strip().lower()
            if ans == "y":
                resume_from = ckpt
            else:
                print("[train] Starting fresh.")

    remaining_steps = args.timesteps - (done_steps if resume_from else 0)
    if remaining_steps <= 0:
        print(f"[train] Already completed {done_steps:,} steps. Use --no-resume to retrain.")
        return

    print(f"[train] steps to run: {remaining_steps:,}")

    # --- Per-variant run info ---
    train_start = datetime.now()
    run_info = {
        "algo":        args.algo.upper(),
        "variant":     variant,
        "patch_file":  str(patch_file.relative_to(exp_dir)) if patch_file else None,
        "patch_diff":  patch_diff or None,
        "timestamp":   train_start.isoformat(),
        "timesteps":   args.timesteps,
        "seed":        args.seed,
        "obs_builder": obs_builder.__class__.__name__,
        "reward_fn":   reward_fn.__class__.__name__,
        "hyperparams": sb3_params,
        "duration_s":  None,
    }
    with open(variant_dir / "run_info.yaml", "w") as f:
        yaml.dump(run_info, f, sort_keys=False)

    # --- Environments ---
    train_env = Monitor(
        FlappyBirdEnv(cfg, obs_builder=make_obs_builder(cfg, obs_override),
                      reward_fn=make_reward_fn(cfg, reward_override)),
        filename=str(variant_dir / "monitor.csv"),
    )
    eval_env = Monitor(
        FlappyBirdEnv(cfg, obs_builder=make_obs_builder(cfg, obs_override),
                      reward_fn=make_reward_fn(cfg, reward_override)),
    )

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = str(variant_dir),
        log_path             = str(variant_dir / "eval"),
        eval_freq            = args.eval_freq,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq   = args.save_freq,
        save_path   = str(variant_dir / "checkpoints"),
        name_prefix = args.algo,
        verbose     = 1,
    )

    # --- Model: load from checkpoint or create fresh ---
    AlgoClass = ALGO_CLASSES[args.algo]
    sb3_params.pop("verbose", None)
    params = {k: v for k, v in sb3_params.items()
              if k not in {"reward_fn", "obs_builder"}}

    if resume_from:
        print(f"[train] Resuming from {resume_from.name}")
        model = AlgoClass.load(
            str(resume_from.with_suffix("")),
            env             = train_env,
            tensorboard_log = str(variant_dir / "tensorboard"),
        )
        model.set_env(train_env)
    else:
        params.update(
            env             = train_env,
            seed            = args.seed,
            tensorboard_log = str(variant_dir / "tensorboard"),
            verbose         = 1,
        )
        model = AlgoClass(**params)

    print(f"[train] obs space   : {train_env.observation_space}")
    print(f"[train] action space: {train_env.action_space}")

    # --- Train ---
    model.learn(
        total_timesteps  = remaining_steps,
        callback         = [eval_callback, checkpoint_callback],
        progress_bar     = True,
        reset_num_timesteps = (resume_from is None),
    )

    # --- Save ---
    final_path = str(variant_dir / "final_model")
    model.save(final_path)
    print(f"[train] final model saved → {final_path}.zip")

    # --- Update run_info with duration ---
    duration_s = (datetime.now() - train_start).total_seconds()
    run_info["duration_s"] = round(duration_s, 1)
    with open(variant_dir / "run_info.yaml", "w") as f:
        yaml.dump(run_info, f, sort_keys=False)
    print(f"[train] duration    : {duration_s/60:.1f} min")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
