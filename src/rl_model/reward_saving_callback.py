"""Callback to save the reward curve during training."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from bone_remodeling.src.rl_model.metrics import MetricsContainer

logger = logging.getLogger(__name__)


class RewardSavingCallback(BaseCallback):
    """Callback to collect episode rewards and save a final plot."""

    def __init__(self, metrics: MetricsContainer, out_path: str = "reward_curve.png", verbose: int = 0, smoothing_window: int = 1000) -> None:
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
        self.steps_in_episode: int = 0

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        self.current_episode_reward += reward
        self.steps_in_episode += 1
        done = bool(self.locals["dones"][0])

        if done:
            self.metrics.episode_rewards.append(self.current_episode_reward)
            self.metrics.steps_at_end_of_episode.append(self.steps_in_episode)
            self.current_episode_reward = 0.0
        return True

    def _on_training_end(self) -> None:
        """Calculate the mean reward on an interval. Add the validation data and plot and save the figure."""
        rewards = self.metrics.episode_rewards
        steps = self.metrics.steps_at_end_of_episode
        val_steps = self.metrics.validation_steps
        val_scores = self.metrics.validation_ssim

        if len(rewards) == 0:
            logger.warning("No rewards collected, skipping plot generation.")
            return

        window = self.smoothing_window
        small_window = max(1, window // 10)

        # Scale reward to below 1
        max_reward = max(abs(min(rewards)), abs(max(rewards)))
        rewards = [r / max_reward for r in rewards]

        # Slightly smoothed
        # Moving-average smoothing
        smoothed = None
        if len(rewards) >= small_window:
            slightly_smoothed = np.convolve(
                rewards, np.ones(small_window) / small_window, mode="valid",
            )

        # Moving-average smoothing
        smoothed = None
        if len(rewards) >= window:
            smoothed = np.convolve(
                rewards, np.ones(window) / window, mode="valid",
            )

        plt.figure(figsize=(10, 5))
        if slightly_smoothed is not None:
            plt.plot(steps[small_window - 1:], slightly_smoothed, c='grey', alpha=0.3, label=f"Slightly smoothed reward (window={small_window})")

        if smoothed is not None:
            plt.plot(steps[window - 1:], smoothed, label=f"Smoothed reward (window={window})")

        if len(val_scores) > 0:
            plt.plot(val_steps, val_scores, marker="o", label="Validation SSIM")

        plt.xlabel("Episode index")
        plt.ylabel("Reward / SSIM")
        plt.title("Training Reward and Validation Performance")
        plt.legend()
        plt.grid(visible=True)
        plt.tight_layout()
        plt.savefig(self.out_path)
        plt.close()

        if self.verbose:
            logger.info(f"Saved reward plot to {self.out_path}")

if __name__ == "__main__":
    #test the plotting function
    import random

    metrics = MetricsContainer()
    for i in range(0, 500_000, 25):
        metrics.steps_at_end_of_episode.append(i)
        metrics.episode_rewards.append(random.uniform(-100, 100))
    metrics.validation_steps = [0, 100_000, 200_000, 300_000, 400_000, 500_000]
    metrics.validation_ssim = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    callback = RewardSavingCallback(metrics, out_path="/home/gijs/Desktop/Thesis/data/figures/test_reward_curve.png", smoothing_window=1000, verbose=1)
    callback._on_training_end()
