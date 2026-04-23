"""Shared MLP policy/value networks and base algorithm class."""
from __future__ import annotations
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_mlp(obs_dim: int, out_dim: int) -> nn.Sequential:
    """2-layer MLP with tanh activations — matches SB3 MlpPolicy default."""
    net = nn.Sequential(
        nn.Linear(obs_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, out_dim),
    )
    # Orthogonal init — same as SB3
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
    return net.to(DEVICE)


def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=DEVICE)


class BaseAlgorithm:
    """Shared interface: predict, save, load, set_env."""

    def __init__(self, env, seed: int = 0, tensorboard_log: str | None = None,
                 verbose: int = 1, **kwargs):
        self.env             = env
        self.seed            = seed
        self.tensorboard_log = tensorboard_log
        self.verbose         = verbose
        self.num_timesteps   = 0
        # Accept and ignore policy= kwarg passed by train.py
        kwargs.pop("policy", None)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        raise NotImplementedError

    def learn(self, total_timesteps: int, callback=None,
              progress_bar: bool = False, reset_num_timesteps: bool = True):
        raise NotImplementedError

    def set_env(self, env):
        self.env = env

    # ------------------------------------------------------------------
    # Save / Load  (ZIP containing JSON hyperparams + torch state dict)
    # ------------------------------------------------------------------

    def _hyperparams(self) -> dict:
        raise NotImplementedError

    def save(self, path: str) -> None:
        if not path.endswith(".zip"):
            path = path + ".zip"
        buf = io.BytesIO()
        torch.save(self._state_dict(), buf)
        buf.seek(0)
        meta = {
            "algo":          self.__class__.__name__,
            "num_timesteps": self.num_timesteps,
            "hyperparams":   self._hyperparams(),
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json",   json.dumps(meta))
            zf.writestr("weights.pt",  buf.read())

    @classmethod
    def load(cls, path: str, env=None, tensorboard_log: str | None = None, **kwargs):
        if not path.endswith(".zip"):
            path = path + ".zip"
        with zipfile.ZipFile(path, "r") as zf:
            meta    = json.loads(zf.read("meta.json"))
            weights = io.BytesIO(zf.read("weights.pt"))

        hp = meta["hyperparams"]
        if tensorboard_log is not None:
            hp["tensorboard_log"] = tensorboard_log
        hp["env"]     = env
        hp["verbose"] = hp.get("verbose", 1)

        model = cls(**hp)
        model._load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
        model.num_timesteps = meta["num_timesteps"]
        return model

    def _state_dict(self) -> dict:
        raise NotImplementedError

    def _load_state_dict(self, sd: dict) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Callback helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_callback(callback):
        from .callbacks import CallbackList, BaseCallback
        if callback is None:
            return BaseCallback()
        if isinstance(callback, list):
            return CallbackList(callback)
        return callback

    @staticmethod
    def _progress(total: int):
        try:
            from tqdm import tqdm
            return tqdm(total=total, unit="step")
        except ImportError:
            return None
