#!/usr/bin/env python3
# flappy.py
"""
Flappy RL — Experiment Manager CLI

Usage:
    python flappy.py

Commands:
    help                        — list all commands
    ls                          — list all experiments with algo completion status
    show exp <exp>              — full details of an experiment + all algo results
    stats exp <exp> [algo]      — training curves (all algos or one)
    eval <exp> <algo>           — launch visual eval for a specific algo run
    train <exp> <algo>          — train an algo into an experiment
    retrain <exp> <algo>        — archive algo run and retrain from scratch
    exit                        — exit
"""

from __future__ import annotations
import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import yaml

RUNS_DIR   = Path("runs")
ARCHIVE_DIR = Path("runs/_archive")
ALGOS       = ["ppo", "dqn", "a2c"]

EXP_CONFIGS = {
    "baseline":       "config/env_baseline.yaml",
    "foam":           "config/env_foam.yaml",
    "foam_kits":      "config/env_foam_kits.yaml",
    "foam_kits_wind": "config/env_foam_kits_wind.yaml",
    "full":           "config/env_full.yaml",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _all_exps() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [r for r in RUNS_DIR.iterdir()
         if r.is_dir() and not r.name.startswith("_")],
        key=lambda p: p.name,
    )


def _find_exp(name: str) -> Path | None:
    target = RUNS_DIR / name
    if target.exists():
        return target
    matches = [r for r in _all_exps() if r.name.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"  Ambiguous '{name}': {[r.name for r in matches]}")
    return None


def _load_exp_meta(exp_dir: Path) -> dict:
    path = exp_dir / "experiment.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_run_info(algo_dir: Path) -> dict:
    path = algo_dir / "run_info.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _monitor_stats(algo_dir: Path) -> dict:
    monitor = algo_dir / "monitor.csv"
    if not monitor.exists():
        return {}
    rewards = []
    try:
        with open(monitor) as f:
            for line in f.readlines()[2:]:
                parts = line.strip().split(",")
                try:
                    rewards.append(float(parts[0]))
                except (ValueError, IndexError):
                    pass
    except Exception:
        return {}
    if not rewards:
        return {}
    last50 = rewards[-50:]
    return {
        "episodes":    len(rewards),
        "best_reward": max(rewards),
        "last50_mean": sum(last50) / len(last50),
        "last50_max":  max(last50),
        "rewards":     rewards,
    }


def _best_checkpoint(algo_dir: Path) -> str | None:
    best = algo_dir / "best_model.zip"
    if best.exists():
        return str(algo_dir / "best_model")
    ckpt_dir = algo_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return str(ckpts[-1].with_suffix(""))
    return None


def _algo_status(exp_dir: Path) -> dict[str, str]:
    """Returns status per algo: 'done', 'partial', or '-'"""
    status = {}
    for algo in ALGOS:
        algo_dir = exp_dir / algo
        if not algo_dir.exists():
            status[algo] = "-"
        elif (algo_dir / "best_model.zip").exists():
            status[algo] = "done"
        elif (algo_dir / "monitor.csv").exists():
            status[algo] = "partial"
        else:
            status[algo] = "-"
    return status


def _env_feature_lines(features: dict) -> list[str]:
    lines = []
    pipe_mode = "Pipe variants: HARD/SOFT/BRITTLE/FOAM" \
        if features.get("pipe_variants") else "Hard pipes only"
    lines.append(f"Pipes          : {pipe_mode}")
    if features.get("pipe_variants"):
        pw = features.get("pipe_weights", {})
        lines.append(f"  Weights      : HARD={pw.get('hard',0):.0%}  SOFT={pw.get('soft',0):.0%}  "
                     f"BRITTLE={pw.get('brittle',0):.0%}  FOAM={pw.get('foam',0):.0%}")
        gh = features.get("gap_heights", {})
        lines.append(f"  Gap heights  : HARD={gh.get('hard',160)}px  SOFT={gh.get('soft',160)}px  "
                     f"BRITTLE={gh.get('brittle',160)}px  FOAM={gh.get('foam',160)}px")
    dmg = "Gradient (near-gap=max, far=min)" \
        if features.get("gradient_damage") else "Flat"
    lines.append(f"Damage         : {dmg}")
    lines.append(f"Bullets        : {'Enabled (' + str(features.get('bullet_count',10)) + '/episode)' if features.get('bullets') else 'Disabled'}")
    lines.append(f"Health kits    : {'Enabled' if features.get('health_kits') else 'Disabled'}")
    lines.append(f"Wind           : {'Enabled' if features.get('wind') else 'Disabled'}")
    lines.append(f"Passive drain  : {features.get('passive_drain', 0.0)} hp/frame")
    lines.append(f"Scroll speed   : {features.get('scroll_speed', 3.0)} px/frame")
    return lines


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_help():
    print("""
  Commands
  ─────────────────────────────────────────────────
  help                        Show this message
  ls                          List all experiments
  show exp <exp>              Full details + all algo results
  stats exp <exp> [algo]      Training curves
  eval <exp> <algo>           Launch visual eval
  train <exp> <algo>          Train an algo into an experiment
  retrain <exp> <algo>        Archive and retrain
  exit                        Exit

  Experiments : baseline | foam | foam_kits | foam_kits_wind | full
  Algorithms  : ppo | dqn | a2c
""")


def cmd_ls():
    exps = _all_exps()
    if not exps:
        print("  No experiments found. Use 'train <exp> <algo>' to start one.")
        return

    print(f"\n  {'EXPERIMENT':<20} {'PPO':^10} {'DQN':^10} {'A2C':^10}")
    print("  " + "─" * 52)

    for exp_dir in exps:
        status = _algo_status(exp_dir)
        symbols = {
            "done":    "✓",
            "partial": "~",
            "-":       "─",
        }
        ppo = symbols[status["ppo"]]
        dqn = symbols[status["dqn"]]
        a2c = symbols[status["a2c"]]

        # Show best reward for completed algos
        def _best(algo):
            s = _monitor_stats(exp_dir / algo)
            return f"{s['best_reward']:.0f}" if s else ""

        ppo_r = _best("ppo") if status["ppo"] == "done" else ""
        dqn_r = _best("dqn") if status["dqn"] == "done" else ""
        a2c_r = _best("a2c") if status["a2c"] == "done" else ""

        ppo_col = f"{ppo} {ppo_r:<7}" if ppo_r else f"{ppo}"
        dqn_col = f"{dqn} {dqn_r:<7}" if dqn_r else f"{dqn}"
        a2c_col = f"{a2c} {a2c_r:<7}" if a2c_r else f"{a2c}"

        print(f"  {exp_dir.name:<20} {ppo_col:^10} {dqn_col:^10} {a2c_col:^10}")

    print(f"\n  ✓ done  ~ partial  ─ not started   (best reward shown for done)")


def cmd_show(exp_name: str):
    exp_dir = _find_exp(exp_name)
    if not exp_dir:
        print(f"  Experiment '{exp_name}' not found.")
        return

    meta = _load_exp_meta(exp_dir)
    features = meta.get("features", {})

    print(f"\n  ┌─ Experiment: {exp_dir.name}")
    print(f"  │  Config source : {meta.get('config_file', '?')}")
    print(f"  │")
    print(f"  │  Environment Features")
    print(f"  │  {'─'*40}")
    for line in _env_feature_lines(features):
        print(f"  │    {line}")

    print(f"  │")
    print(f"  │  Algorithm Runs")
    print(f"  │  {'─'*40}")

    for algo in ALGOS:
        algo_dir = exp_dir / algo
        info     = _load_run_info(algo_dir)
        stats    = _monitor_stats(algo_dir)
        ckpt     = _best_checkpoint(algo_dir)

        if not info:
            print(f"  │  [{algo.upper():<3}]  not started")
            continue

        ts    = info.get("timestamp", "?")[:19]
        steps = info.get("timesteps", "?")
        obs   = info.get("obs_builder", "?")
        rf    = info.get("reward_fn", "?")
        ckpt_label = "best_model" if ckpt and "best_model" in ckpt else (Path(ckpt).name if ckpt else "none")

        print(f"  │  [{algo.upper():<3}]  {ts}  |  {steps:,} steps  |  {obs}  |  {rf}")
        if stats:
            print(f"  │         episodes={stats['episodes']}  best={stats['best_reward']:.1f}"
                  f"  last50_mean={stats['last50_mean']:.1f}  checkpoint={ckpt_label}")
        else:
            print(f"  │         no training data yet")

    print(f"  └{'─'*60}")


def cmd_stats(exp_name: str, algo_filter: str | None = None):
    exp_dir = _find_exp(exp_name)
    if not exp_dir:
        print(f"  Experiment '{exp_name}' not found.")
        return

    algos_to_show = [algo_filter] if algo_filter else ALGOS

    for algo in algos_to_show:
        algo_dir = exp_dir / algo
        stats    = _monitor_stats(algo_dir)
        if not stats:
            print(f"  [{algo.upper()}]  no data")
            continue

        rewards   = stats["rewards"]
        n_buckets = 50
        bucket_sz = max(len(rewards) // n_buckets, 1)
        buckets   = []
        for i in range(0, len(rewards), bucket_sz):
            chunk = rewards[i:i+bucket_sz]
            buckets.append(sum(chunk) / len(chunk))

        max_val = max(buckets)
        min_val = min(buckets)
        chart_h = 8
        width   = len(buckets)

        print(f"\n  [{algo.upper()}]  {exp_dir.name}  —  "
              f"episodes={len(rewards)}  best={stats['best_reward']:.1f}  "
              f"last50_mean={stats['last50_mean']:.1f}")
        print(f"  {max_val:>8.0f} ┐")
        for row in range(chart_h, 0, -1):
            threshold = min_val + (max_val - min_val) * row / chart_h
            bar = "".join("█" if b >= threshold else " " for b in buckets)
            print(f"  {'':>9}│{bar}")
        print(f"  {min_val:>8.0f} └" + "─" * width)
        print(f"  {'':>9}  ep 0{' '*(width-10)}ep {len(rewards)}")


def cmd_eval(exp_name: str, algo: str):
    exp_dir = _find_exp(exp_name)
    if not exp_dir:
        print(f"  Experiment '{exp_name}' not found.")
        return
    if algo not in ALGOS:
        print(f"  Unknown algo '{algo}'. Choose from: {ALGOS}")
        return

    algo_dir = exp_dir / algo
    ckpt     = _best_checkpoint(algo_dir)
    if not ckpt:
        print(f"  No checkpoint found for {exp_name}/{algo}.")
        return

    ckpt_label = "best_model" if (algo_dir / "best_model.zip").exists() else Path(ckpt).name
    print(f"  Launching eval: {exp_name}/{algo}  [{ckpt_label}]")
    subprocess.run([sys.executable, "evals.py", "--exp", exp_name, "--algo", algo])


def cmd_train(exp_name: str, algo: str):
    if algo not in ALGOS:
        print(f"  Unknown algo '{algo}'. Choose from: {ALGOS}")
        return

    # Check if this algo run already exists
    algo_dir = RUNS_DIR / exp_name / algo
    if algo_dir.exists() and (algo_dir / "monitor.csv").exists():
        print(f"  Run {exp_name}/{algo} already exists. Use 'retrain {exp_name} {algo}' to restart.")
        return

    # Validate experiment name
    if exp_name not in EXP_CONFIGS and not (RUNS_DIR / exp_name / "env_config.yaml").exists():
        print(f"  Unknown experiment '{exp_name}'. Known: {list(EXP_CONFIGS.keys())}")
        return

    print(f"  Training: {exp_name} / {algo.upper()}")
    print(f"  (Press Ctrl+C to stop)\n")
    subprocess.run([sys.executable, "train.py", "--exp", exp_name, "--algo", algo])


def cmd_retrain(exp_name: str, algo: str):
    exp_dir  = _find_exp(exp_name)
    if not exp_dir:
        print(f"  Experiment '{exp_name}' not found.")
        return
    if algo not in ALGOS:
        print(f"  Unknown algo '{algo}'. Choose from: {ALGOS}")
        return

    algo_dir = exp_dir / algo
    if not algo_dir.exists():
        print(f"  No existing run for {exp_name}/{algo}. Use 'train {exp_name} {algo}' instead.")
        return

    # Archive the algo subfolder only
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = ARCHIVE_DIR / f"{exp_name}_{algo}_{ts}"
    shutil.copytree(algo_dir, archive_path)
    shutil.rmtree(algo_dir)
    print(f"  Archived → {archive_path}")

    print(f"  Retraining: {exp_name} / {algo.upper()}")
    print(f"  (Press Ctrl+C to stop)\n")
    subprocess.run([sys.executable, "train.py", "--exp", exp_name, "--algo", algo])


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def run_shell():
    print("\n  Flappy RL — Experiment Manager")
    print("  Type 'help' for commands, 'exit' to quit.\n")

    while True:
        try:
            raw = input("  flappy> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd == "exit":
            print("  Bye.")
            break
        elif cmd == "help":
            cmd_help()
        elif cmd == "ls":
            cmd_ls()
        elif cmd == "show" and len(parts) >= 3 and parts[1] == "exp":
            cmd_show(parts[2])
        elif cmd == "stats" and len(parts) >= 3 and parts[1] == "exp":
            algo_filter = parts[3] if len(parts) >= 4 else None
            cmd_stats(parts[2], algo_filter)
        elif cmd == "eval" and len(parts) >= 3:
            cmd_eval(parts[1], parts[2])
        elif cmd == "train" and len(parts) >= 3:
            cmd_train(parts[1], parts[2])
        elif cmd == "retrain" and len(parts) >= 3:
            cmd_retrain(parts[1], parts[2])
        else:
            print(f"  Unknown command: '{raw}'  (type 'help' for commands)")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    run_shell()
