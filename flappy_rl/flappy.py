#!/usr/bin/env python3
# flappy.py
"""
Flappy RL — Experiment Manager CLI

Usage:
    python flappy.py

Commands:
    help                            list all commands
    status                          show partial runs and pending suggestions
    ls                              list all experiments with algo completion grid
    show exp <exp>                  full details of an experiment + all variant results
    stats exp <exp> [variant]       training curves (all variants or one)
    compare <exp> <algo>            compare all variants for one algo side by side
    eval <exp> <variant>            launch visual eval for a specific variant
    train <exp> <algo> [variant]    train (or resume) a variant
    analyze <exp> <variant>         diagnose results and suggest a patch
    approve                         run the pending suggestion
    reject [reason]                 discard the pending suggestion
    retrain <exp> <variant>         archive variant and retrain from scratch
    exit                            exit
"""

from __future__ import annotations
import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import yaml

RUNS_DIR    = Path("runs")
ARCHIVE_DIR = Path("runs/_archive")
ALGOS       = ["ppo", "dqn", "a2c"]

EXP_CONFIGS = {
    "baseline":       "config/env_baseline.yaml",
    "foam":           "config/env_foam.yaml",
    "foam_kits":      "config/env_foam_kits.yaml",
    "foam_kits_wind": "config/env_foam_kits_wind.yaml",
    "full":           "config/env_full.yaml",
}

# Pending suggestion state file (one per experiment)
SUGGESTION_FILE = "pending_suggestion.yaml"


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


def _load_run_info(variant_dir: Path) -> dict:
    path = variant_dir / "run_info.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _monitor_stats(variant_dir: Path) -> dict:
    monitor = variant_dir / "monitor.csv"
    if not monitor.exists():
        return {}
    rewards, lengths = [], []
    try:
        with open(monitor) as f:
            for line in f.readlines()[2:]:
                parts = line.strip().split(",")
                try:
                    rewards.append(float(parts[0]))
                    lengths.append(float(parts[1]))
                except (ValueError, IndexError):
                    pass
    except Exception:
        return {}
    if not rewards:
        return {}
    last50r = rewards[-50:]
    last50l = lengths[-50:]
    return {
        "episodes":       len(rewards),
        "best_reward":    max(rewards),
        "last50_mean":    sum(last50r) / len(last50r),
        "last50_max":     max(last50r),
        "last50_ep_len":  sum(last50l) / len(last50l),
        "rewards":        rewards,
        "lengths":        lengths,
    }


def _best_checkpoint(variant_dir: Path) -> str | None:
    best = variant_dir / "best_model.zip"
    if best.exists():
        return str(variant_dir / "best_model")
    ckpt_dir = variant_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return str(ckpts[-1].with_suffix(""))
    return None


def _variants_for_exp(exp_dir: Path) -> list[Path]:
    """All variant subdirs (excludes patches/, _archive/, experiment.yaml etc.)"""
    skip = {"patches", "_archive"}
    return sorted([
        d for d in exp_dir.iterdir()
        if d.is_dir() and d.name not in skip and not d.name.startswith("_")
    ], key=lambda p: p.name)


def _variant_algo(variant_dir: Path) -> str | None:
    info = _load_run_info(variant_dir)
    return info.get("algo", "").lower() or None


def _variant_status(variant_dir: Path) -> str:
    if not variant_dir.exists():
        return "-"
    if (variant_dir / "final_model.zip").exists():
        return "done"
    if (variant_dir / "monitor.csv").exists():
        return "partial"
    return "-"


def _algo_variants(exp_dir: Path, algo: str) -> list[Path]:
    """All variant dirs whose run_info says they belong to this algo."""
    result = []
    for v in _variants_for_exp(exp_dir):
        info = _load_run_info(v)
        if info.get("algo", "").lower() == algo:
            result.append(v)
    return result


def _env_feature_lines(features: dict) -> list[str]:
    lines = []
    pipe_mode = "Pipe variants: HARD/SOFT/BRITTLE/FOAM" \
        if features.get("pipe_variants") else "Hard pipes only"
    lines.append(f"Pipes          : {pipe_mode}")
    if features.get("pipe_variants"):
        pw = features.get("pipe_weights", {})
        lines.append(f"  Weights      : HARD={pw.get('hard',0):.0%}  SOFT={pw.get('soft',0):.0%}  "
                     f"BRITTLE={pw.get('brittle',0):.0%}  FOAM={pw.get('foam',0):.0%}")
    dmg = "Gradient (near-gap=max, far=min)" if features.get("gradient_damage") else "Flat"
    lines.append(f"Damage         : {dmg}")
    bc  = features.get("bullet_count", 10)
    lines.append(f"Bullets        : {'Enabled (' + str(bc) + '/episode)' if features.get('bullets') else 'Disabled'}")
    lines.append(f"Health kits    : {'Enabled' if features.get('health_kits') else 'Disabled'}")
    lines.append(f"Wind           : {'Enabled' if features.get('wind') else 'Disabled'}")
    lines.append(f"Passive drain  : {features.get('passive_drain', 0.0)} hp/frame")
    lines.append(f"Scroll speed   : {features.get('scroll_speed', 3.0)} px/frame")
    return lines


def _ascii_curve(rewards: list[float], n_buckets: int = 50, height: int = 8) -> list[str]:
    bucket_sz = max(len(rewards) // n_buckets, 1)
    buckets   = []
    for i in range(0, len(rewards), bucket_sz):
        chunk = rewards[i:i+bucket_sz]
        buckets.append(sum(chunk) / len(chunk))
    max_val = max(buckets)
    min_val = min(buckets)
    width   = len(buckets)
    lines   = [f"  {max_val:>8.0f} ┐"]
    for row in range(height, 0, -1):
        threshold = min_val + (max_val - min_val) * row / height
        bar = "".join("█" if b >= threshold else " " for b in buckets)
        lines.append(f"  {'':>9}│{bar}")
    lines.append(f"  {min_val:>8.0f} └" + "─" * width)
    lines.append(f"  {'':>9}  ep 0{' '*(width-10)}ep {len(rewards)}")
    return lines


# ---------------------------------------------------------------------------
# Pending suggestion helpers
# ---------------------------------------------------------------------------

def _suggestion_path(exp_dir: Path) -> Path:
    return exp_dir / SUGGESTION_FILE


def _load_suggestion(exp_dir: Path) -> dict | None:
    path = _suggestion_path(exp_dir)
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _save_suggestion(exp_dir: Path, suggestion: dict) -> None:
    with open(_suggestion_path(exp_dir), "w") as f:
        yaml.dump(suggestion, f, sort_keys=False)


def _clear_suggestion(exp_dir: Path) -> None:
    path = _suggestion_path(exp_dir)
    if path.exists():
        path.unlink()


def _all_pending_suggestions() -> list[tuple[Path, dict]]:
    result = []
    for exp_dir in _all_exps():
        s = _load_suggestion(exp_dir)
        if s:
            result.append((exp_dir, s))
    return result


def _all_partial_runs() -> list[tuple[Path, Path]]:
    """Returns (exp_dir, variant_dir) for all partial runs."""
    result = []
    for exp_dir in _all_exps():
        for v in _variants_for_exp(exp_dir):
            if _variant_status(v) == "partial":
                result.append((exp_dir, v))
    return result


# ---------------------------------------------------------------------------
# Claude CLI resolver
# ---------------------------------------------------------------------------

def _find_claude() -> str | None:
    """Return path to the claude binary, or None if not found."""
    import shutil, glob as _glob
    # 1. On PATH
    path = shutil.which("claude")
    if path:
        return path
    # 2. VSCode extension bundle (common on Linux/Mac)
    patterns = [
        os.path.expanduser("~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"),
        os.path.expanduser("~/.cursor/extensions/anthropic.claude-code-*/resources/native-binary/claude"),
        "/usr/local/bin/claude",
    ]
    for pattern in patterns:
        matches = _glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]   # latest version
    return None


# ---------------------------------------------------------------------------
# Analysis engine (Claude-powered)
# ---------------------------------------------------------------------------

def _next_variant_name(exp_dir: Path, variant_name: str) -> str:
    """Increment the numeric suffix: ppo → ppo2, ppo2 → ppo3."""
    base = variant_name.rstrip("0123456789")
    nums = [int(v.name[len(base):]) for v in _variants_for_exp(exp_dir)
            if v.name.startswith(base) and v.name[len(base):].isdigit()]
    next_num = (max(nums) + 1) if nums else 2
    return f"{base}{next_num}"


def _build_claude_prompt(exp_name: str, variant_name: str,
                         stats: dict, info: dict, features: dict) -> str:
    rewards   = stats["rewards"]
    episodes  = stats["episodes"]
    bucket_sz = max(episodes // 40, 1)
    buckets   = []
    for i in range(0, len(rewards), bucket_sz):
        chunk = rewards[i:i+bucket_sz]
        buckets.append(sum(chunk) / len(chunk))
    curve_str = "\n".join(
        f"  ep {i*bucket_sz:>5}: {b:>8.1f}" for i, b in enumerate(buckets)
    )

    algo             = info.get("algo", "?")
    obs_builder      = info.get("obs_builder", "?")
    reward_fn        = info.get("reward_fn", "?")
    hyperparams      = info.get("hyperparams", {})
    feature_items    = "\n".join(f"  {k}: {v}" for k, v in features.items())
    hyperparam_items = "\n".join(f"  {k}: {v}" for k, v in hyperparams.items())

    # Valid patch keys Claude is allowed to suggest
    patch_keys = """
Allowed patch keys (stable-baselines3 hyperparams + special overrides):
  PPO/A2C : learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda,
             clip_range, ent_coef, vf_coef, max_grad_norm
  DQN     : learning_rate, buffer_size, learning_starts, batch_size, gamma,
             train_freq, gradient_steps, target_update_interval,
             exploration_fraction, exploration_final_eps
  Special : reward_fn (one of: survival, health_aware, scored)
            timesteps (total training steps, e.g. 2000000)
"""

    return f"""You are analyzing a deep RL training run for a modified Flappy Bird environment.

## Experiment
- Name: {exp_name}
- Variant: {variant_name}
- Algorithm: {algo}
- Obs builder: {obs_builder}
- Reward function: {reward_fn}

## Environment features
{feature_items}

## Hyperparameters used
{hyperparam_items}

## Training statistics
- Total episodes : {episodes}
- Best reward    : {stats['best_reward']:.1f}
- Last-50 mean   : {stats['last50_mean']:.1f}
- Last-50 ep len : {stats['last50_ep_len']:.1f} frames

## Reward curve (bucketed mean per ~{bucket_sz} episodes)
{curve_str}

## Environment context
The bird navigates through pipes. Base reward: +1/frame alive, -100 on death.
Pipe types: HARD (instant death on collision), SOFT/BRITTLE/FOAM (gradient damage —
  near gap edge = max damage, far from gap = min damage, never instant death).
HealthAwareReward: damage penalty scales inversely with current health (low health = bigger penalty).
ScoredReward: +pipe_bonus per pipe cleared, to give DQN denser signal.
Observation: bird_y, vel_y, next 2 pipes (dist, gap_top, gap_bot, type), health, bullets.
Action: Discrete(4) — bit0=flap, bit1=shoot. Shooting destroys next non-hard pipe.

## Your task
1. In 3–5 sentences: diagnose what the reward curve shows and why it happened.
   Be specific — reference the algorithm, reward function, and environment.
2. Suggest hyperparameter changes to fix it.

{patch_keys}
End your response with a fenced yaml block containing ONLY the suggested overrides.
Use only keys from the allowed list above. No text after the yaml block.

Example format:
```yaml
ent_coef: 0.01
learning_rate: 0.0001
```"""


def _parse_patch_from_output(output: str) -> dict:
    """Extract the last ```yaml ... ``` block from Claude's output and parse it."""
    import re
    blocks = re.findall(r"```yaml\s*(.*?)```", output, re.DOTALL)
    if not blocks:
        return {}
    try:
        patch = yaml.safe_load(blocks[-1].strip())
        return patch if isinstance(patch, dict) else {}
    except yaml.YAMLError:
        return {}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_help():
    print("""
  Commands
  ──────────────────────────────────────────────────────────
  help                          Show this message
  status                        Partial runs + pending suggestions
  ls                            List all experiments
  show exp <exp>                Full details + all variant results
  stats exp <exp> [variant]     Training curves
  compare <exp> <algo>          All variants for one algo side by side
  eval <exp> <variant>          Launch visual eval
  train <exp> <algo> [variant]  Train or resume a variant
  analyze <exp> <variant>       Diagnose and suggest next patch
  approve                       Run the pending suggestion
  reject [reason]               Discard the pending suggestion
  retrain <exp> <variant>       Archive and retrain a variant
  exit                          Exit

  Experiments : baseline | foam | foam_kits | foam_kits_wind | full
  Algorithms  : ppo | dqn | a2c
""")


def cmd_status():
    partial    = _all_partial_runs()
    pending    = _all_pending_suggestions()

    if not partial and not pending:
        print("  All clear — no partial runs or pending suggestions.")
        return

    if partial:
        print(f"\n  Partial runs ({len(partial)}):")
        for exp_dir, v in partial:
            info = _load_run_info(v)
            ts   = info.get("timestamp", "?")[:16]
            print(f"    {exp_dir.name}/{v.name}  [{ts}]  — resume with: train {exp_dir.name} {info.get('algo','?').lower()} {v.name}")

    if pending:
        print(f"\n  Pending suggestions ({len(pending)}):")
        for exp_dir, s in pending:
            print(f"    {exp_dir.name}: suggest '{s['next_variant']}' from '{s['source']}'")
            print(f"      patch: {s['patch_diff']}")
            print(f"      type 'approve' to run, 'reject' to discard")


def cmd_ls():
    exps = _all_exps()
    if not exps:
        print("  No experiments found. Use 'train <exp> <algo>' to start one.")
        return

    print(f"\n  {'EXPERIMENT':<20} {'PPO':<22} {'DQN':<22} {'A2C':<22}")
    print("  " + "─" * 88)

    for exp_dir in exps:
        algo_cols = []
        for algo in ALGOS:
            variants = _algo_variants(exp_dir, algo)
            if not variants:
                algo_cols.append(f"{'─':^22}")
                continue
            parts = []
            for v in variants:
                st    = _variant_status(v)
                sym   = {"done": "✓", "partial": "~", "-": "─"}[st]
                stats = _monitor_stats(v) if st == "done" else {}
                br    = f" {stats['best_reward']:.0f}" if stats else ""
                parts.append(f"{sym}{v.name}{br}")
            algo_cols.append(("  ".join(parts))[:22])

        pending = _load_suggestion(exp_dir)
        flag    = " ★" if pending else ""
        print(f"  {exp_dir.name + flag:<20} {algo_cols[0]:<22} {algo_cols[1]:<22} {algo_cols[2]:<22}")

    print(f"\n  ✓ done  ~ partial  ─ not started  ★ pending suggestion")


def cmd_show(exp_name: str):
    exp_dir = _find_exp(exp_name)
    if not exp_dir:
        print(f"  Experiment '{exp_name}' not found.")
        return

    meta     = _load_exp_meta(exp_dir)
    features = meta.get("features", {})

    print(f"\n  ┌─ Experiment: {exp_dir.name}")
    print(f"  │  Config source : {meta.get('config_file', '?')}")
    print(f"  │")
    print(f"  │  Environment Features")
    print(f"  │  {'─'*45}")
    for line in _env_feature_lines(features):
        print(f"  │    {line}")

    print(f"  │")
    print(f"  │  Variants")
    print(f"  │  {'─'*45}")

    variants = _variants_for_exp(exp_dir)
    if not variants:
        print(f"  │    No runs yet.")
    for v in variants:
        info  = _load_run_info(v)
        stats = _monitor_stats(v)
        ckpt  = _best_checkpoint(v)
        if not info:
            continue

        algo       = info.get("algo", "?")
        ts         = info.get("timestamp", "?")[:19]
        steps      = info.get("timesteps", "?")
        dur        = info.get("duration_s")
        dur_str    = f"{dur/60:.1f} min" if dur else "in progress"
        obs        = info.get("obs_builder", "?")
        rf         = info.get("reward_fn", "?")
        patch_diff = info.get("patch_diff")
        ckpt_label = "best_model" if ckpt and "best_model" in ckpt else (Path(ckpt).name if ckpt else "none")
        status     = _variant_status(v)
        sym        = {"done": "✓", "partial": "~", "-": "─"}[status]

        print(f"  │  {sym} [{v.name}]  {algo}  {ts}  {dur_str}")
        if patch_diff:
            print(f"  │     patch: {patch_diff}")
        print(f"  │     obs={obs}  reward={rf}  steps={steps:,}  ckpt={ckpt_label}")
        if stats:
            print(f"  │     episodes={stats['episodes']}  best={stats['best_reward']:.1f}  "
                  f"last50_mean={stats['last50_mean']:.1f}  last50_ep_len={stats['last50_ep_len']:.0f}")

    # Pending suggestion
    suggestion = _load_suggestion(exp_dir)
    if suggestion:
        print(f"  │")
        print(f"  │  ★ Pending suggestion: {suggestion['next_variant']}")
        print(f"  │    patch: {suggestion['patch_diff']}")
        print(f"  │    type 'approve' to run or 'reject' to discard")

    print(f"  └{'─'*60}")


def cmd_stats(exp_name: str, variant_filter: str | None = None):
    exp_dir = _find_exp(exp_name)
    if not exp_dir:
        print(f"  Experiment '{exp_name}' not found.")
        return

    variants = _variants_for_exp(exp_dir)
    if variant_filter:
        variants = [v for v in variants if v.name == variant_filter]

    for v in variants:
        stats = _monitor_stats(v)
        if not stats:
            print(f"  [{v.name}]  no data")
            continue
        info = _load_run_info(v)
        algo = info.get("algo", "?")
        print(f"\n  [{v.name}]  {algo}  —  "
              f"episodes={stats['episodes']}  best={stats['best_reward']:.1f}  "
              f"last50_mean={stats['last50_mean']:.1f}")
        for line in _ascii_curve(stats["rewards"]):
            print(line)


def cmd_compare(exp_name: str, algo: str):
    exp_dir = _find_exp(exp_name)
    if not exp_dir:
        print(f"  Experiment '{exp_name}' not found.")
        return
    if algo not in ALGOS:
        print(f"  Unknown algo '{algo}'. Choose from: {ALGOS}")
        return

    variants = _algo_variants(exp_dir, algo)
    if not variants:
        print(f"  No {algo.upper()} variants found for '{exp_name}'.")
        return

    print(f"\n  Comparing {algo.upper()} variants for: {exp_name}")
    print(f"  {'─'*60}")
    print(f"  {'VARIANT':<20} {'EPISODES':>9} {'BEST RW':>9} {'L50 MEAN':>9} {'L50 EP LEN':>11} {'DURATION':>10}")
    print(f"  {'─'*60}")

    for v in variants:
        stats     = _monitor_stats(v)
        info      = _load_run_info(v)
        dur       = info.get("duration_s")
        dur_str   = f"{dur/60:.1f}m" if dur else "?"
        patch_diff = info.get("patch_diff")
        ep        = f"{stats['episodes']}"       if stats else "-"
        br        = f"{stats['best_reward']:.1f}" if stats else "-"
        l50       = f"{stats['last50_mean']:.1f}" if stats else "-"
        l50l      = f"{stats['last50_ep_len']:.0f}" if stats else "-"
        print(f"  {v.name:<20} {ep:>9} {br:>9} {l50:>9} {l50l:>11} {dur_str:>10}")
        if patch_diff:
            print(f"  {'':20}   patch: {patch_diff}")

    print()
    # Show curves stacked
    for v in variants:
        stats = _monitor_stats(v)
        if not stats:
            continue
        print(f"  [{v.name}]")
        for line in _ascii_curve(stats["rewards"], height=6):
            print(line)
        print()


def cmd_eval(exp_name: str, variant: str):
    exp_dir = _find_exp(exp_name)
    if not exp_dir:
        print(f"  Experiment '{exp_name}' not found.")
        return

    variant_dir = exp_dir / variant
    ckpt        = _best_checkpoint(variant_dir)
    if not ckpt:
        print(f"  No checkpoint found for {exp_name}/{variant}.")
        return

    ckpt_label = "best_model" if (variant_dir / "best_model.zip").exists() else Path(ckpt).name
    print(f"  Launching eval: {exp_name}/{variant}  [{ckpt_label}]")
    subprocess.run([sys.executable, "evals.py", "--exp", exp_name, "--variant", variant])


def cmd_train(exp_name: str, algo: str, variant: str | None = None):
    if algo not in ALGOS:
        print(f"  Unknown algo '{algo}'. Choose from: {ALGOS}")
        return
    if exp_name not in EXP_CONFIGS and not (RUNS_DIR / exp_name / "env_config.yaml").exists():
        print(f"  Unknown experiment '{exp_name}'. Known: {list(EXP_CONFIGS.keys())}")
        return

    cmd = [sys.executable, "train.py", "--exp", exp_name, "--algo", algo]
    if variant:
        cmd += ["--variant", variant]
    print(f"  Training: {exp_name} / {variant or algo}  (Press Ctrl+C to stop)\n")
    subprocess.run(cmd)


def cmd_analyze(exp_name: str, variant_name: str):
    exp_dir = _find_exp(exp_name)
    if not exp_dir:
        print(f"  Experiment '{exp_name}' not found.")
        return

    variant_dir = exp_dir / variant_name
    stats       = _monitor_stats(variant_dir)
    info        = _load_run_info(variant_dir)

    if not stats or not info:
        print(f"  Not enough data to analyze '{exp_name}/{variant_name}'.")
        return

    claude_bin = _find_claude()
    if not claude_bin:
        print("  Claude CLI not found — cannot analyze.")
        print("  Install claude via: npm install -g @anthropic-ai/claude-code")
        return

    meta         = _load_exp_meta(exp_dir)
    features     = meta.get("features", {})
    algo         = info.get("algo", "ppo").lower()
    next_variant = _next_variant_name(exp_dir, variant_name)
    prompt       = _build_claude_prompt(exp_name, variant_name, stats, info, features)

    print(f"\n  [ANALYZE]  {exp_name} / {variant_name}")
    print(f"  {'─'*55}\n")

    # Capture output so we can parse the yaml block, but also stream to terminal
    result = subprocess.run(
        [claude_bin, "-p", prompt],
        capture_output=True,
        text=True,
    )
    output = result.stdout
    print(output)

    # Parse patch from Claude's yaml block
    patch = _parse_patch_from_output(output)

    if not patch:
        print("  (no yaml patch block found in response — nothing to approve)")
        return

    print(f"  {'─'*55}")
    print(f"  Patch extracted:")
    for k, v in patch.items():
        print(f"    {k}: {v}")
    print(f"\n  Next variant : {next_variant}")
    print(f"  Train with   : train {exp_name} {algo} {next_variant}")
    print(f"\n  Type 'approve' to create patch + train, or 'reject' to discard.")

    suggestion = {
        "exp_name":     exp_name,
        "source":       variant_name,
        "algo":         algo,
        "next_variant": next_variant,
        "patch_file":   f"patches/{next_variant}.yaml",
        "patch_diff":   patch,
        "timestamp":    datetime.now().isoformat(),
    }
    _save_suggestion(exp_dir, suggestion)


def cmd_approve():
    pending = _all_pending_suggestions()
    if not pending:
        print("  No pending suggestions.")
        return
    if len(pending) > 1:
        print(f"  Multiple pending suggestions:")
        for exp_dir, s in pending:
            print(f"    {exp_dir.name}: {s['next_variant']}")
        print("  Run 'reject' on others first, or clear manually.")
        return

    exp_dir, suggestion = pending[0]
    exp_name    = suggestion["exp_name"]
    algo        = suggestion["algo"]
    next_variant = suggestion["next_variant"]
    patch_file  = exp_dir / suggestion["patch_file"]
    patch_diff  = suggestion["patch_diff"]

    # Write the patch file
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    with open(patch_file, "w") as f:
        yaml.dump(patch_diff, f, sort_keys=False)
    print(f"  Patch written → {patch_file}")

    _clear_suggestion(exp_dir)
    cmd_train(exp_name, algo, next_variant)


def cmd_reject(reason: str | None = None):
    pending = _all_pending_suggestions()
    if not pending:
        print("  No pending suggestions to reject.")
        return
    for exp_dir, s in pending:
        _clear_suggestion(exp_dir)
        msg = f" Reason: {reason}" if reason else ""
        print(f"  Rejected suggestion for {exp_dir.name}/{s['next_variant']}.{msg}")


def cmd_retrain(exp_name: str, variant_name: str):
    exp_dir = _find_exp(exp_name)
    if not exp_dir:
        print(f"  Experiment '{exp_name}' not found.")
        return

    variant_dir = exp_dir / variant_name
    if not variant_dir.exists():
        print(f"  Variant '{variant_name}' not found in '{exp_name}'.")
        return

    info = _load_run_info(variant_dir)
    algo = info.get("algo", "").lower()

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = ARCHIVE_DIR / f"{exp_name}_{variant_name}_{ts}"
    shutil.copytree(variant_dir, archive_path)
    shutil.rmtree(variant_dir)
    print(f"  Archived → {archive_path}")

    cmd_train(exp_name, algo, variant_name)


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def run_shell():
    print("\n  Flappy RL — Experiment Manager")
    print("  Type 'help' for commands, 'exit' to quit.\n")

    # Greet with any pending state
    cmd_status()

    while True:
        try:
            raw = input("\n  flappy> ").strip()
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
        elif cmd == "status":
            cmd_status()
        elif cmd == "ls":
            cmd_ls()
        elif cmd == "show" and len(parts) >= 3 and parts[1] == "exp":
            cmd_show(parts[2])
        elif cmd == "stats" and len(parts) >= 3 and parts[1] == "exp":
            vf = parts[3] if len(parts) >= 4 else None
            cmd_stats(parts[2], vf)
        elif cmd == "compare" and len(parts) >= 3:
            cmd_compare(parts[1], parts[2])
        elif cmd == "eval" and len(parts) >= 3:
            cmd_eval(parts[1], parts[2])
        elif cmd == "train" and len(parts) >= 3:
            variant = parts[3] if len(parts) >= 4 else None
            cmd_train(parts[1], parts[2], variant)
        elif cmd == "analyze" and len(parts) >= 3:
            cmd_analyze(parts[1], parts[2])
        elif cmd == "approve":
            cmd_approve()
        elif cmd == "reject":
            reason = " ".join(parts[1:]) if len(parts) > 1 else None
            cmd_reject(reason)
        elif cmd == "retrain" and len(parts) >= 3:
            cmd_retrain(parts[1], parts[2])
        else:
            print(f"  Unknown command: '{raw}'  (type 'help' for commands)")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    run_shell()
