# plot_training.py
"""
Plot mean episodic reward over training timesteps from monitor.csv logs.
Works for any run that used SB3's Monitor wrapper.

Usage:
    python plot_training.py                         # all runs in runs/
    python plot_training.py --algo ppo              # only PPO runs
    python plot_training.py --algo ppo --algo dqn   # PPO and DQN
    python plot_training.py --runs runs/ppo_001 runs/ppo_002  # specific runs
    python plot_training.py --window 20             # wider smoothing
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_monitor(csv_path: Path) -> pd.DataFrame:
    """
    SB3 Monitor writes a one-line JSON header then CSV rows:
        #{"t_start": ..., "env_id": ...}
        r,l,t
        12.0,45,1.23
    """
    with open(csv_path) as f:
        lines = f.readlines()

    # Skip the comment header line(s)
    data_lines = [l for l in lines if not l.startswith("#")]
    if len(data_lines) < 2:
        return pd.DataFrame()   # no episodes logged yet

    from io import StringIO
    df = pd.read_csv(StringIO("".join(data_lines)))
    # r = episode reward, l = episode length, t = wall time
    df.rename(columns={"r": "reward", "l": "length", "t": "time"}, inplace=True)
    df["timestep"] = df["length"].cumsum()
    return df


def find_runs(runs_dir: Path, algos: list[str] | None) -> list[Path]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory at '{runs_dir}'")
    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()],
                  key=lambda r: r.stat().st_mtime)
    if algos:
        runs = [r for r in runs if any(r.name.startswith(a) for a in algos)]
    return runs


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (x_indices, smoothed_values) with valid convolution."""
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    x = np.arange(window - 1, len(values))
    return x, smoothed


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

ALGO_COLORS = {
    "ppo": "#4C8BF5",
    "dqn": "#E67E22",
    "a2c": "#2ECC71",
}
DEFAULT_COLOR = "#888888"


def plot_training_curves(
    runs: list[Path],
    window: int,
    save_path: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Mean episodic reward during training", fontsize=13)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward")

    plotted = 0
    for run in runs:
        csv = run / "monitor.csv"
        if not csv.exists():
            print(f"[plot] no monitor.csv in {run.name}, skipping")
            continue

        df = load_monitor(csv)
        if df.empty:
            print(f"[plot] no episodes yet in {run.name}, skipping")
            continue

        # Detect algo for colour
        algo = next((a for a in ALGO_COLORS if run.name.startswith(a)), None)
        color = ALGO_COLORS.get(algo, DEFAULT_COLOR)

        rewards   = df["reward"].values
        timesteps = df["timestep"].values

        # Raw (faint)
        ax.plot(timesteps, rewards, alpha=0.15, color=color, linewidth=0.7)

        # Smoothed
        if len(rewards) >= window:
            xi, ys = rolling_mean(rewards, window)
            ax.plot(timesteps[xi], ys, color=color, linewidth=2,
                    label=f"{run.name}  (n={len(rewards)} eps)")
        else:
            ax.plot(timesteps, rewards, color=color, linewidth=2,
                    label=f"{run.name}  (n={len(rewards)} eps, no smoothing)")

        plotted += 1

    if plotted == 0:
        print("[plot] nothing to plot — no monitor.csv files found.")
        return

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] saved → {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Args + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs",   nargs="*", default=None,
                   help="Explicit run folder paths. Defaults to all runs/ subfolders.")
    p.add_argument("--algo",   nargs="*", dest="algos", default=None,
                   choices=["ppo", "dqn", "a2c"],
                   help="Filter by algorithm prefix.")
    p.add_argument("--window", type=int, default=10,
                   help="Rolling mean window in episodes (default: 10)")
    p.add_argument("--save",   default=None,
                   help="Save plot to this path (e.g. comparison.png)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.runs:
        runs = [Path(r) for r in args.runs]
    else:
        runs = find_runs(Path("runs"), args.algos)

    print(f"[plot] found {len(runs)} run(s): {[r.name for r in runs]}")

    save_path = Path(args.save) if args.save else None
    plot_training_curves(runs, window=args.window, save_path=save_path)


if __name__ == "__main__":
    main()