# Flappy Bird RL

CS5180 project comparing PPO, A2C, and DQN on a modified Flappy Bird environment with health mechanics and pipe variants. All algorithms are implemented from scratch in PyTorch.

## Project Structure

```
flappy_rl/
├── train.py                  # main training script
├── evals.py                  # evaluate a saved model
├── plot_training.py          # plot monitor.csv training curves
├── algorithms/               # PPO, A2C, DQN implementations
│   ├── ppo.py
│   ├── a2c.py
│   ├── dqn.py
│   ├── callbacks.py          # EvalCallback, CheckpointCallback
│   └── monitor.py            # Monitor wrapper (writes monitor.csv)
├── envs/                     # Flappy Bird environment
│   ├── flappy_env.py
│   ├── rewards.py            # SurvivalReward, ScoredReward, ExponentialHealthReward, ...
│   └── config.py
├── config/                   # environment YAML configs
│   ├── config1.yaml          # Config 1: hard pipes only, 8D obs
│   └── config2.yaml          # Config 2: pipe variants + health, 12D obs
└── runs/                     # experiment outputs (auto-created)
    ├── baseline/             # Config 1 experiments
    │   ├── patches/          # hyperparameter overrides per variant
    │   └── <variant>/        # per-variant outputs
    │       ├── monitor.csv   # per-episode reward/length/time
    │       ├── evaluations.npz
    │       └── checkpoints/
    └── foam/                 # Config 2 experiments
        └── patches/
```

## Setup

```bash
source ~/py3/bin/activate
cd flappy_rl
```

All commands below are run from `flappy_rl/`.

## Training

### Basic usage

```bash
# PPO on baseline env (Config 1), 1M steps
python train.py --exp baseline --algo ppo

# A2C on baseline env
python train.py --exp baseline --algo a2c

# DQN on baseline env
python train.py --exp baseline --algo dqn
```

`--exp` controls which experiment folder under `runs/` is used and which env config is loaded. `--algo` sets the algorithm. The variant name defaults to the algo name.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--exp` | — | Experiment name (maps to `runs/<exp>/`) |
| `--algo` | `ppo` | Algorithm: `ppo`, `a2c`, `dqn` |
| `--variant` | algo name | Output subfolder name, e.g. `ppo2` |
| `--patch` | auto | Path to patch YAML (relative to exp dir) |
| `--timesteps` | 1,000,000 | Total training steps |
| `--seed` | 42 | Random seed |
| `--eval-freq` | 10,000 | How often to run evaluation episodes |

### Using a patch file

Patches are YAML files that override hyperparameters for a specific variant. They live in `runs/<exp>/patches/`.

```bash
# Run the tuned PPO variant (ppo2) with its patch
python train.py --exp baseline --algo ppo --variant ppo2 --patch patches/ppo2.yaml

# Auto-resolve: if variant name matches a patch filename, patch is applied automatically
python train.py --exp baseline --algo ppo --variant ppo2
```

Example patch (`runs/baseline/patches/ppo2.yaml`):
```yaml
learning_rate: 0.0001
n_epochs: 4
n_steps: 4096
ent_coef: 0.01
timesteps: 3000000
```

Any key not in the patch keeps its algorithm default. Special keys `reward_fn` and `timesteps` are handled separately (not passed as algorithm hyperparams).

### Foam environment experiments

```bash
# PPO with exponential health reward
python train.py --exp foam --algo ppo --variant ppo_exponential --patch patches/ppo_exponential.yaml

# PPO clip range ablation
python train.py --exp foam --algo ppo --variant ppo_clip05 --patch patches/ppo_clip05.yaml
python train.py --exp foam --algo ppo --variant ppo_clip10 --patch patches/ppo_clip10.yaml

# A2C on foam env
python train.py --exp foam --algo a2c --variant a2c_exponential --patch patches/a2c_exponential.yaml

# A2C with long rollout (ablation)
python train.py --exp foam --algo a2c --variant a2c_long_rollout --patch patches/a2c_long_rollout.yaml
```

### Resume training

If a checkpoint exists in `runs/<exp>/<variant>/checkpoints/`, training resumes automatically from the latest checkpoint.

## Evaluation

```bash
# Evaluate a trained variant (3 episodes, deterministic policy)
python evals.py --exp baseline --variant ppo2

# More episodes
python evals.py --exp baseline --variant ppo2 --episodes 10

# Test behavior at low health (foam env)
python evals.py --exp foam --variant ppo_exponential --health 20.0

# Test with a specific foam pipe fraction
python evals.py --exp foam --variant ppo_exponential --foam-pct 0.8
```

The model is loaded from `runs/<exp>/<variant>/checkpoints/` (latest checkpoint).

## Analyzing Training Curves

### Plot monitor.csv

`monitor.csv` records every episode: reward (`r`), length in steps (`l`), and wall time (`t`).

```bash
# Plot all variants in an experiment
python plot_training.py --exp baseline

# Compare specific algos
python plot_training.py --exp baseline --algo ppo --algo a2c

# Wider smoothing window
python plot_training.py --exp baseline --window 20
```

### Read monitor.csv directly with pandas

```python
import pandas as pd

df = pd.read_csv("runs/baseline/ppo2/monitor.csv", skiprows=1)
# columns: r (reward), l (episode length), t (wall time)

print(df["r"].describe())         # reward stats
print(df["r"].rolling(50).mean()) # smoothed reward
```

### Compare algorithms

```python
import pandas as pd
import matplotlib.pyplot as plt

algos = {"PPO": "runs/baseline/ppo2", "A2C": "runs/baseline/a2c5", "DQN": "runs/baseline/dqn"}

for name, path in algos.items():
    df = pd.read_csv(f"{path}/monitor.csv", skiprows=1)
    df["cumsteps"] = df["l"].cumsum()
    smoothed = df["r"].rolling(50, min_periods=1).mean()
    plt.plot(df["cumsteps"], smoothed, label=name)

plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.legend()
plt.savefig("comparison.png")
```

## Experiments Summary

| Experiment | Config | Variants |
|-----------|--------|---------|
| `baseline` | Config 1 (hard pipes, 8D obs) | `ppo`, `ppo2`, `a2c`, `a2c5`, `dqn` |
| `foam` | Config 2 (pipe variants + health, 12D obs) | `ppo2`, `ppo_exponential`, `ppo_asymmetric`, `ppo_clip05`, `ppo_clip10`, `a2c_exponential`, `a2c_long_rollout` |

### Reward functions

| Name | Description |
|------|-------------|
| `survival` | +1/frame survived |
| `scored` | +10 per pipe cleared |
| `exponential` | exponential urgency scaling by health |
| `threshold` | flat penalty below health threshold |
| `asymmetric` | survival + foam graze bonus + hard pipe penalty |

Set via patch YAML: `reward_fn: exponential`

### Environment configs

| Config | File | Description |
|--------|------|-------------|
| Config 1 | `config/config1.yaml` | Hard pipes only, no health drain |
| Config 2 | `config/config2.yaml` | Mixed pipe types (hard/soft/foam), health system, 12D obs |
