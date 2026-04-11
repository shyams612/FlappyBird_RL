# train.py
"""
Unified training entry point for PPO, DQN, and A2C.

Usage:
    python train.py                                 # PPO, default config, 1M steps
    python train.py --algo dqn
    python train.py --algo a2c
    python train.py --algo ppo --timesteps 2000000
    python train.py --algo dqn --config config/config1.yaml --run-name dqn_config1

Outputs (all under runs/<run_name>/):
    best_model.zip      — best checkpoint by eval reward
    final_model.zip     — model at end of training
    checkpoints/        — periodic saves
    tensorboard/        — TensorBoard logs

Compare all runs:
    tensorboard --logdir runs/
"""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.flappy_env import FlappyBirdEnv
from envs.config import EnvConfig


# ---------------------------------------------------------------------------
# Per-algorithm hyperparameters
# Keep these separate so each algo can be tuned independently.
# All values are SB3 defaults adjusted for this environment.
# ---------------------------------------------------------------------------

ALGO_PARAMS = {
    "ppo": dict(
        policy        = "MlpPolicy",
        learning_rate = 3e-4,
        n_steps       = 2048,       # steps collected per update
        batch_size    = 64,
        n_epochs      = 10,         # gradient steps per update
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        verbose       = 1,
    ),
    "dqn": dict(
        policy               = "MlpPolicy",
        learning_rate        = 1e-4,
        buffer_size          = 100_000,   # replay buffer
        learning_starts      = 10_000,    # steps before first gradient update
        batch_size           = 64,
        tau                  = 1.0,       # target network update rate
        gamma                = 0.99,
        train_freq           = 4,         # update every N steps
        gradient_steps       = 1,
        target_update_interval = 1000,    # hard update target net every N steps
        exploration_fraction = 0.1,       # fraction of training spent exploring
        exploration_final_eps= 0.05,
        verbose              = 1,
    ),
    "a2c": dict(
        policy        = "MlpPolicy",
        learning_rate = 7e-4,
        n_steps       = 5,          # shorter rollouts than PPO
        gamma         = 0.99,
        gae_lambda    = 1.0,
        ent_coef      = 0.0,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        verbose       = 1,
    ),
}

ALGO_CLASSES = {"ppo": PPO, "dqn": DQN, "a2c": A2C}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo",       choices=["ppo", "dqn", "a2c"], default="ppo")
    p.add_argument("--config",     default=None,  help="Path to YAML env config")
    p.add_argument("--timesteps",  type=int,      default=1_000_000)
    p.add_argument("--run-name",   default=None,  help="Override auto-generated run name")
    p.add_argument("--seed",       type=int,      default=42)
    p.add_argument("--eval-freq",  type=int,      default=10_000)
    p.add_argument("--save-freq",  type=int,      default=100_000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Config ---
    cfg = EnvConfig.from_yaml(args.config) if args.config else EnvConfig()

    # --- Run directory ---
    run_name = args.run_name or f"{args.algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir  = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] algo      : {args.algo.upper()}")
    print(f"[train] run dir   : {run_dir}")
    print(f"[train] timesteps : {args.timesteps:,}")

    # --- Environments ---
    train_env = Monitor(FlappyBirdEnv(cfg), filename=str(run_dir / "monitor.csv"))
    eval_env  = Monitor(FlappyBirdEnv(cfg))

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = str(run_dir),
        log_path             = str(run_dir / "eval"),
        eval_freq            = args.eval_freq,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq   = args.save_freq,
        save_path   = str(run_dir / "checkpoints"),
        name_prefix = args.algo,
        verbose     = 1,
    )

    # --- Model ---
    AlgoClass = ALGO_CLASSES[args.algo]
    params    = ALGO_PARAMS[args.algo].copy()
    params.update(
        env             = train_env,
        seed            = args.seed,
        tensorboard_log = str(run_dir / "tensorboard"),
    )
    model = AlgoClass(**params)

    print(f"[train] obs space : {train_env.observation_space}")
    print(f"[train] action space: {train_env.action_space}")

    # --- Train ---
    model.learn(
        total_timesteps = args.timesteps,
        callback        = [eval_callback, checkpoint_callback],
        progress_bar    = True,
    )

    # --- Save ---
    final_path = str(run_dir / "final_model")
    model.save(final_path)
    print(f"[train] final model saved → {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()