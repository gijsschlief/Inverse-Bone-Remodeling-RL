"""Training environment for reinforcement learning in bone remodeling simulation."""

import logging
import time

import numpy as np
import torch
from bone_remodeling.rl_model.reward_calculation import calculate_similarity
from bone_remodeling.surrogate_model.loader import load_surrogate_model
from bone_remodeling.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.surrogate_model.normalizor import (
    normalize_data,
    unnormalize_data,
)


class ReinforcementLearningEnvironment:
    """Environment for reinforcement learning in bone remodeling simulation."""

    def __init__(self, model_path: str) -> None:
        """Initialize the environment with a surrogate model."""
        surrogate_model_and_normalization_params = load_surrogate_model(model_path, ReversedSurrogateModel)
        if surrogate_model_and_normalization_params is None:
            raise ValueError("Failed to load the surrogate model.")
        self.surrogate_model, self.x_mean, self.x_std, self.y_mean, self.y_std = surrogate_model_and_normalization_params

        logging.info(f"Normalization parameters loaded: x_mean={self.x_mean.shape}, x_std={self.x_std.shape}, y_mean={self.y_mean.shape}, y_std={self.y_std.shape}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.surrogate_model.to(self.device)
        self.surrogate_model.eval()

    def _surrogate_model_forward(self, force_profile: np.ndarray, return_shape: tuple) -> np.ndarray:
        """Forward pass through the surrogate model.

        Args:
        ----
            force_profile (np.ndarray): The force profile applied to the bone.
            return_shape (tuple): The shape to return the predicted density.

        Returns:
        -------
            np.ndarray: The predicted density from the surrogate model.

        """
        if self.x_mean is not None:
            force_profile, _, _, _, _ = normalize_data(force_profile, None, None, self.x_mean, self.x_std)

        with torch.no_grad():
            force_profile_tensor = torch.from_numpy(force_profile.reshape(1, -1).astype(np.float32)).to(self.device)

        density_tensor: torch.Tensor = self.surrogate_model(force_profile_tensor)
        surrogate_density = density_tensor.detach().cpu().numpy()

        if self.y_mean is not None:
            surrogate_density = unnormalize_data(surrogate_density, self.y_mean, self.y_std)
        return surrogate_density.reshape(return_shape)

    def reward(self, force_profile: np.ndarray, target_density: np.ndarray) -> float:
        """Calculate the reward based on the force profile and target density.

        The reward is calculated as the negative mean squared error between the predicted and target densities.

        Args:
        ----
            force_profile (np.ndarray): The force profile applied to the bone.
            target_density (np.ndarray): The target density of the bone.

        Returns:
        -------
            float: The calculated reward.

        """
        surrogate_density = self._surrogate_model_forward(force_profile, target_density.shape)

        reward_score = calculate_similarity(
            reference_matrix=target_density,
            comparison_matrix=surrogate_density,
            method="mse",
            baseline=0.1,
            threshold=0.5
            )
        return reward_score

def main() -> None:
    """Demonstrates the environment and reward calculation."""
    model_path = "/home/gijs/Desktop/Thesis/data/models/trained_model_1.pth"
    training_environment = ReinforcementLearningEnvironment(model_path)

    # Example force profile and target density
    force_profile = np.random.rand(3, 10)  # Example force profile
    target_density = np.random.rand(10, 10)  # Example target density

    logging.info("Calculating reward...")
    start_time = time.time()
    reward = training_environment.reward(force_profile, target_density)
    end_time = time.time()

    logging.info(f"Reward calculation took {end_time - start_time:.4f} seconds")
    logging.info(f"Calculated reward: {reward}")


if __name__ == "__main__":
    main()
