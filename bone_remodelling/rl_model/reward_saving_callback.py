"""Callback to save the reward curve during training."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from bone_remodelling.rl_model.metrics import MetricsContainer

logger = logging.getLogger(__name__)


class RewardSavingCallback(BaseCallback):
    """Callback to collect episode rewards and save a final plot."""

    def __init__(self, metrics: MetricsContainer, out_path: Path, verbose: int = 0, smoothing_window: int = 1000) -> None:
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
        self.current_episode_reward = 0.0
        self.out_path = out_path
        self.smoothing_window = smoothing_window

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])

        self.current_episode_reward += reward

        if done:
            # Store episode reward
            self.metrics.episode_rewards.append(self.current_episode_reward)
            self.metrics.episode_indices.append(len(self.metrics.episode_rewards) - 1)
            self.metrics.episode_end_timesteps.append(self.num_timesteps)

            self.current_episode_reward = 0.0

        return True

    def _on_training_end(self) -> None:
        rewards = self.metrics.episode_rewards
        episode_x = np.array(self.metrics.episode_end_timesteps)
        val_steps = self.metrics.validation_steps
        val_scores = self.metrics.validation_ssim

        if len(rewards) == 0:
            logger.warning("No rewards collected, skipping plot.")
            return

        # Smooth reward
        window = self.smoothing_window
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            smoothed_x = episode_x[window - 1:]
        else:
            smoothed = None

        # Small smooth reward
        small_window = self.smoothing_window // 10
        if len(rewards) >= small_window:
            small_smoothed = np.convolve(rewards, np.ones(small_window) / small_window, mode="valid")
            small_smoothed_x = episode_x[small_window - 1:]
        else:
            small_smoothed = None

        fig, reward_axis = plt.subplots(figsize=(12, 5))

        # Reward curve (left y-axis)
        if small_smoothed is not None:
            reward_axis.plot(small_smoothed_x, small_smoothed, alpha=0.3, label=f"Smoothed reward (w={small_window})", color="gray")
        if smoothed is not None:
            reward_axis.plot(smoothed_x, smoothed, label=f"Smoothed reward (w={window})")

        reward_axis.set_xlabel("Episode index")
        reward_axis.set_ylabel("Reward")
        reward_axis.legend(loc="upper left")
        reward_axis.grid(visible=True)

        # Validation curve (right y-axis)
        if len(val_scores) > 0:
            ax2 = reward_axis.twinx()
            ax2.plot(val_steps, val_scores, "o-", color="orange", label="Validation SSIM")
            ax2.set_ylabel("SSIM")
            ax2.legend(loc="upper right")

        plt.title("Training Reward and Validation Performance")
        plt.tight_layout()
        plt.savefig(self.out_path)
        plt.close()

        if self.verbose:
            logger.info(f"Saved reward plot to {self.out_path}")

if __name__ == "__main__":
    #test the plotting function
    import random

    metrics = MetricsContainer()
    figure_path = Path(__file__).parent.parent.parent / Path("data", "figures", "test_reward_curve.png")
    for i in range(0, 500_000, 25):
        metrics.episode_indices.append(i // 25)
        metrics.episode_end_timesteps.append(i)
        metrics.episode_rewards.append(random.uniform(-100, 100))
    metrics.validation_steps = [0, 100_000, 200_000, 300_000, 400_000, 500_000]
    metrics.validation_ssim = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    callback = RewardSavingCallback(metrics, out_path=figure_path, smoothing_window=1000, verbose=1)
    callback._on_training_end()
