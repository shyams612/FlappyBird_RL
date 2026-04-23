"""EvalCallback and CheckpointCallback matching the SB3 interface."""
from __future__ import annotations
import numpy as np


class BaseCallback:
    def __init__(self):
        self.model = None

    def init_callback(self, model):
        self.model = model

    def on_step(self) -> bool:
        return True

    def on_rollout_end(self):
        pass

    def on_training_end(self):
        pass


class CallbackList(BaseCallback):
    def __init__(self, callbacks: list):
        super().__init__()
        self.callbacks = callbacks

    def init_callback(self, model):
        self.model = model
        for c in self.callbacks:
            c.init_callback(model)

    def on_step(self) -> bool:
        return all(c.on_step() for c in self.callbacks)

    def on_rollout_end(self):
        for c in self.callbacks:
            c.on_rollout_end()

    def on_training_end(self):
        for c in self.callbacks:
            c.on_training_end()


class EvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        best_model_save_path: str,
        log_path: str,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        self.eval_env           = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path           = log_path
        self.eval_freq          = eval_freq
        self.n_eval_episodes    = n_eval_episodes
        self.deterministic      = deterministic
        self.verbose            = verbose

        self._best_mean_reward  = -np.inf
        self._all_timesteps: list[int]   = []
        self._all_results:   list[list]  = []
        self._all_ep_lengths: list[list] = []

    def on_step(self) -> bool:
        if self.model.num_timesteps % self.eval_freq != 0:
            return True

        rewards, lengths = [], []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_r, ep_l = 0.0, 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, r, terminated, truncated, _ = self.eval_env.step(int(action))
                ep_r += r
                ep_l += 1
                done = terminated or truncated
            rewards.append(ep_r)
            lengths.append(ep_l)

        mean_r = float(np.mean(rewards))
        self._all_timesteps.append(self.model.num_timesteps)
        self._all_results.append(rewards)
        self._all_ep_lengths.append(lengths)

        # Save evaluations.npz — matches SB3 format consumed by generate_plots.py
        import os
        os.makedirs(self.log_path, exist_ok=True)
        np.savez(
            f"{self.log_path}/evaluations",
            timesteps  = np.array(self._all_timesteps),
            results    = np.array(self._all_results),
            ep_lengths = np.array(self._all_ep_lengths),
        )

        if mean_r > self._best_mean_reward:
            self._best_mean_reward = mean_r
            if self.best_model_save_path:
                import os
                os.makedirs(self.best_model_save_path, exist_ok=True)
                self.model.save(f"{self.best_model_save_path}/best_model")

        if self.verbose >= 1:
            print(f"  [eval] ts={self.model.num_timesteps:,}  "
                  f"mean_r={mean_r:.1f}  best={self._best_mean_reward:.1f}")
        return True


class CheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "model",
        verbose: int = 0,
    ):
        super().__init__()
        self.save_freq   = save_freq
        self.save_path   = save_path
        self.name_prefix = name_prefix
        self.verbose     = verbose

    def on_step(self) -> bool:
        if self.model.num_timesteps % self.save_freq == 0:
            import os
            os.makedirs(self.save_path, exist_ok=True)
            path = f"{self.save_path}/{self.name_prefix}_{self.model.num_timesteps}_steps"
            self.model.save(path)
            if self.verbose >= 1:
                print(f"  [ckpt] saved → {path}.zip")
        return True
