"""Callback to save the reward curve during training."""

import logging

import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class RewardSavingCallback(BaseCallback):
    """Callback to collect episode rewards and save a final plot."""

    def __init__(self, out_path: str = "reward_curve.png", verbose: int = 0) -> None:
        """Initialize the reward saving callback.

        Args:
        ----
            out_path (str): Path to save the reward plot.
            verbose (int): Verbosity level.

        """
        super().__init__(verbose)
        self.out_path = out_path
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        # accumulate the reward at each step
        self.episode_rewards.append(self.locals["rewards"][0])
        return True

    def _on_training_end(self) -> None:
        """Plot and save the figure."""
        plt.figure(figsize=(8, 4))
        plt.plot(self.episode_rewards)
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.title("Episode Reward over Training")
        plt.tight_layout()
        plt.savefig(self.out_path)
        plt.close()
        if self.verbose:
            logger.info(f"Saved reward plot to {self.out_path}")
