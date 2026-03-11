"""Training environment for reinforcement learning in bone remodeling simulation."""

import logging

import numpy as np
from gymnasium import Env, spaces

from bone_remodelling.rl_model.forward_pass import (
    ForwardPass,
)
from bone_remodelling.rl_model.parameters import RLParameters
from bone_remodelling.rl_model.reward_calculation import calculate_similarity

logger = logging.getLogger(__name__)


class BoneRemodelingEnvironment(Env):
    """Gym Environment for reinforcement learning in bone remodeling simulation."""

    def __init__(
        self,
        forwarder: ForwardPass,
        target_densities: np.ndarray,
        target_forces: np.ndarray,
        rl_parameters: RLParameters,
    ) -> None:
        """Initialize the environment with a surrogate model."""
        super().__init__()
        self.forwarder = forwarder

        self.target_densities = target_densities
        self.density_shape = target_densities[0].shape
        self._profile_length = np.max(self.density_shape)

        self.max_steps = rl_parameters.max_steps
        self.density_constraint = rl_parameters.density_constraint
        self.force_boundary = rl_parameters.force_boundary

        self.force_shape = (3, self._profile_length)
        self.force_profile = np.zeros(
            self.force_shape,
            dtype=np.float32,
        )

        self.target_forces = target_forces
        self.num_samples = len(target_densities)
        self.current_sample_index = 0

        # Define the action space
        self.per_step_force_change = rl_parameters.per_step_force_change
        action_space_lower_bounds = np.array(
            [0.0] * 30 + [-self.per_step_force_change], dtype=np.float32
        )
        action_space_upper_bounds = np.array(
            [1.0] * 30 + [self.per_step_force_change], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=action_space_lower_bounds,
            high=action_space_upper_bounds,
            dtype=np.float32,
        )

        # Define the observation space
        self.grid_size = int(np.prod(self.density_shape))
        observation_space_lower_bounds = np.vstack(
            [
                np.full(self.density_shape, -self.density_constraint, dtype=np.float32),
                np.full(self.force_shape, -self.force_boundary, dtype=np.float32),
            ],
        )
        observation_space_upper_bounds = np.vstack(
            [
                np.full(self.density_shape, self.density_constraint, dtype=np.float32),
                np.full(self.force_shape, self.force_boundary, dtype=np.float32),
            ],
        )
        self.observation_space = spaces.Box(
            low=observation_space_lower_bounds,
            high=observation_space_upper_bounds,
            dtype=np.float32,
        )

        # Episode variables
        self.current_step = 0
        self.last_predicted_density = np.zeros(self.density_shape, dtype=np.float32)
        self.previous_ssim = 0.0
        self.target_density = target_densities[0]
        self.target_force = target_forces[0]
        self.reward = 0.0

    def _get_observation(self) -> np.ndarray:
        density_difference = (self.target_density - self.last_predicted_density).astype(
            np.float32
        )
        return np.vstack([density_difference, self.force_profile.astype(np.float32)])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode.

        Conforms to Gymnasium API: accepts seed/options, returns (obs, info).

        Args:
        ----
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for the reset.

        Returns:
        -------
            tuple: A tuple containing the initial observation and an info dictionary.

        """
        super().reset(seed=seed)
        np.random.seed(seed)

        # Pick a new random sample
        self.current_sample_index = np.random.randint(self.num_samples)
        self.target_density = self.target_densities[self.current_sample_index]
        self.target_force = self.target_forces[self.current_sample_index]

        # Reset predictions
        self.current_step = 0
        self.last_predicted_density = np.zeros(self.density_shape, dtype=np.float32)
        self.force_profile = np.zeros(
            self.force_shape,
            dtype=np.float32,
        )
        initial_ssim = calculate_similarity(
            reference_matrix=self.target_density,
            comparison_matrix=self.last_predicted_density,
            method="ssim",
        )
        self.previous_ssim = initial_ssim

        info: dict = {}
        info["sample_index"] = self.current_sample_index
        info["target_density"] = self.target_density
        info["target_force"] = self.target_force
        return self._get_observation(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Perform a step in the environment.

        Args:
        ----
            action (np.ndarray): The force profile applied to the bone.

        Returns:
        -------
            tuple: A tuple containing the observation, reward, done flag, and additional info.

        """
        location_index = np.argmax(action[:30])
        side_index, peak_position = np.divmod(location_index, self._profile_length)
        self.force_profile[side_index, peak_position] += action[30].item()
        self.force_profile = np.clip(
            self.force_profile,
            -self.force_boundary,
            self.force_boundary,
        )

        predicted_density = self.forwarder.forward_pass(
            force_profile=self.force_profile,
        )
        current_ssim = calculate_similarity(
            reference_matrix=self.target_density,
            comparison_matrix=predicted_density,
            method="ssim",
            baseline=0.1,
            threshold=0.5,
        )

        self.last_predicted_density = predicted_density.astype(np.float32)

        self.current_step += 1
        self.reward = current_ssim - self.previous_ssim
        self.previous_ssim = current_ssim

        # base termination: either out of steps or success
        ssim_threshold = 0.99
        success = current_ssim >= ssim_threshold
        terminated = self.current_step >= self.max_steps or success
        truncated = False

        info = {
            "force_profile": self.force_profile,
            "predicted_density": predicted_density,
            "reward": self.reward,
            "success": success,
            "current_step": self.current_step,
            "sample_index": self.current_sample_index,
        }

        return self._get_observation(), self.reward, terminated, truncated, info

    def get_data_for_visualization(
        self,
    ) -> tuple[
        tuple[int, np.ndarray, np.ndarray],
        tuple[int, np.ndarray, np.ndarray],
        float,
    ]:
        """Pass data needed for rendering the environment to the callback function."""
        sample_information = (
            self.current_sample_index,
            self.target_force,
            self.target_density,
        )
        estimate_information = (
            self.current_step,
            self.force_profile,
            self.last_predicted_density,
        )
        return sample_information, estimate_information, self.reward
