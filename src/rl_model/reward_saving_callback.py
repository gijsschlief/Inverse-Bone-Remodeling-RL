"""Callback to save the reward curve during training."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from bone_remodeling.src.rl_model.metrics import MetricsContainer

logger = logging.getLogger(__name__)


class RewardSavingCallback(BaseCallback):
    """Callback to collect episode rewards and save a final plot."""

    def __init__(self, metrics: MetricsContainer, out_path: str = "reward_curve.png", verbose: int = 0, smoothing_window: int = 50) -> None:
        """Initialize the reward saving callback.

        Args:
        ----
            metrics (MetricsContainer): Container to store training metrics.
            out_path (str): Path to save the reward plot.
            verbose (int): Verbosity level.
            smoothing_window (int): Window size for smoothing the reward curve.

        """
        super().__init__(verbose)
        self.metrics = metrics
        self.current_episode_reward: float = 0.0
        self.out_path = out_path
        self.episode_rewards: list[float] = []
        self.smoothing_window = smoothing_window

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        self.current_episode_reward += reward
        done = bool(self.locals["dones"][0])

        if done:
            self.metrics.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        return True

    def _on_training_end(self) -> None:
        """Calculate the mean reward on an interval. Add the validation data and plot and save the figure."""
        rewards = self.metrics.episode_rewards
        val_steps = self.metrics.validation_steps
        val_scores = self.metrics.validation_ssim

        # smoothing
        if len(rewards) > 0:
            window = self.smoothing_window
            smoothed = (
                np.convolve(rewards, np.ones(window)/window, mode="valid")
                if len(rewards) >= window else rewards
            )

        plt.figure(figsize=(10,5))
        plt.plot(rewards, alpha=0.3, label="Episode Reward (raw)")
        plt.plot(range(window - 1, window - 1 + len(smoothed)), smoothed, label="Smoothed reward")

        if len(val_scores) > 0:
            plt.plot(val_steps, val_scores, marker="o", label="Validation SSIM")

        plt.xlabel("Training steps / episodes")
        plt.ylabel("Reward / SSIM")
        plt.title("Training Reward and Validation Performance")
        plt.legend()
        plt.grid(showgrid=True)
        plt.tight_layout()
        plt.savefig(self.out_path)
        plt.close()
        if self.verbose:
            logger.info(f"Saved reward plot to {self.out_path}")
