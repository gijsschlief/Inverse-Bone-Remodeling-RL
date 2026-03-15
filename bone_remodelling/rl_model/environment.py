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
            [-10.0] * 30 + [-self.per_step_force_change],
            dtype=np.float32,
        )
        action_space_upper_bounds = np.array(
            [10.0] * 30 + [self.per_step_force_change],
            dtype=np.float32,
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
        self.max_curriculum_complexity = 10
        self._build_curriculum_learning_groups()

    def _build_curriculum_learning_groups(self) -> None:
        """Curriculum learning implementation.

        This method organizes the training samples into groups based on their complexity (e.g., number of active forces). It also slowly expands the steps the model can make in line with the number of forces.
        """
        self.groups: dict[int, list[int]] = {
            i: [] for i in range(1, self.max_curriculum_complexity + 1)
        }
        self.current_max_complexity = 1
        threshold_force = 1e-3
        for i, force in enumerate(self.target_forces):
            num_active_points = np.sum(np.any(np.abs(force) > threshold_force, axis=0))
            complexity = min(num_active_points, self.max_curriculum_complexity)
            if complexity == 0:
                complexity = 1
            self.groups[complexity].append(i)

    def increase_curriculum_complexity(self) -> None:
        """Advance the curriculum."""
        if self.current_max_complexity <= self.max_curriculum_complexity:
            self.current_max_complexity += 1
            self.max_steps = self.current_max_complexity * 5
            logger.info(
                f"--- Curriculum Advanced to Complexity {self.current_max_complexity} --- (increasing max steps to {self.max_steps})",
            )

    def set_curriculum_complexity(self, complexity: int) -> None:
        """Set the curriculum complexity level (used when resuming training)."""
        self.current_max_complexity = min(complexity, self.max_curriculum_complexity)
        self.max_steps = self.current_max_complexity * 5
        logger.info(f"Environment complexity synchronized to Level {self.current_max_complexity}")

    def _select_sample(self) -> int:
        """Select a sample index either from the current curriculum learning group or a previous group."""
        sample_from_previous_groups = 0.3
        if self.num_samples == 1:
            return 0

        if (
            np.random.rand() < sample_from_previous_groups
            and self.current_max_complexity > 1
        ):
            complexity = np.random.randint(1, self.current_max_complexity)
            if self.groups[complexity]:
                return np.random.choice(self.groups[complexity])

        if (
            self.current_max_complexity <= self.max_curriculum_complexity
            and self.groups[self.current_max_complexity]
        ):
            return np.random.choice(self.groups[self.current_max_complexity])

        return np.random.randint(0, self.num_samples)

    def _update_sample(self) -> None:
        """Update the current sample index and corresponding target density and force."""
        self.current_sample_index = self._select_sample()
        self.target_density = self.target_densities[self.current_sample_index]
        self.target_force = self.target_forces[self.current_sample_index]

    def _get_observation(self) -> np.ndarray:
        density_difference = (self.target_density - self.last_predicted_density).astype(
            np.float32,
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

        self._update_sample()

        # Reset predictions
        self.current_step = 0
        self.last_predicted_density = np.zeros(self.density_shape, dtype=np.float32)
        self.force_profile = np.zeros(
            self.force_shape,
            dtype=np.float32,
        )

        _, _ = self._calculate_reward(self.last_predicted_density)

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
        location_logits = action[:30]
        magnitude_change = action[30].item()

        exp_logits = np.exp(
            location_logits - np.max(location_logits),
        )  # Subtract max for stability
        probabilities = exp_logits / exp_logits.sum()

        prob_reshaped = probabilities.reshape(self.force_shape)
        self.force_profile += magnitude_change * prob_reshaped

        self.force_profile = np.clip(
            self.force_profile,
            -self.force_boundary,
            self.force_boundary,
        )

        predicted_density = self.forwarder.forward_pass(
            force_profile=self.force_profile,
        )

        self.last_predicted_density = predicted_density.astype(np.float32)
        self.current_step += 1
        self.reward, current_ssim = self._calculate_reward(predicted_density)

        # base termination: either out of steps or success
        ssim_threshold = 0.95
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

    def _calculate_reward(self, predicted_density: np.ndarray) -> tuple[float, float]:
        """Calculate the reward based on the change in SSIM."""
        current_ssim = calculate_similarity(
            reference_matrix=self.target_density,
            comparison_matrix=predicted_density,
            method="ssim",
            baseline=0.1,
            threshold=0.5,
        )
        epsilon = 1e-6
        reward = -np.log(1 - current_ssim + epsilon) - (
            -np.log(1 - self.previous_ssim + epsilon)
        )
        self.previous_ssim = current_ssim
        return reward, current_ssim

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
