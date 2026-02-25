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
        self.per_step_dead_zone = rl_parameters.per_step_dead_zone
        self.per_step_location_change = rl_parameters.per_step_location_change

        self.action_space = spaces.Box(
            low=np.array([-self.per_step_force_change, -self.per_step_dead_zone, -self.per_step_location_change], dtype=np.float32),
            high=np.array([self.per_step_force_change, self.per_step_dead_zone, self.per_step_location_change], dtype=np.float32),
            dtype=np.float32,
        )

        self.peak_magnitude = np.float32(0.0)
        self.peak_location = np.int8(0)

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
        self.current_step = 0
        self.reward = 0.0
        self.peak_magnitude = np.float32(0.0)
        self.peak_location = np.int8(0)

        # Pick a new random sample
        self.current_sample_index = np.random.randint(self.num_samples)

        # Update target for this episode
        self.target_density = self.target_densities[self.current_sample_index]
        self.target_force = self.target_forces[self.current_sample_index]

        # reset observation
        self.last_predicted_density = np.zeros(self.density_shape, dtype=np.float32)
        difference = (
            self.target_density.astype(np.float32) - self.last_predicted_density
        ).astype(np.float32)
        episode_observation = np.vstack(
            [difference, np.zeros(self.force_profile.shape, dtype=np.float32)],
        )
        info: dict = {}
        info["sample_index"] = self.current_sample_index
        info["target_density"] = self.target_density
        info["target_force"] = self.target_force
        return episode_observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Perform a step in the environment.

        Args:
        ----
            action (np.ndarray): The force profile applied to the bone.

        Returns:
        -------
            tuple: A tuple containing the observation, reward, done flag, and additional info.

        """
        peak_action, dead_zone, location_action = action
        self.peak_magnitude = self.peak_magnitude + np.float32(peak_action)
        if self.peak_magnitude < -self.force_boundary:
            self.peak_magnitude = np.float32(-self.force_boundary)
        elif self.peak_magnitude > self.force_boundary:
            self.peak_magnitude = np.float32(self.force_boundary)

        move_peak = np.int8(0) if dead_zone > 0 else np.int8(np.sign(location_action))
        self.peak_location = np.mod(
            self.peak_location + move_peak,
            self._profile_length * 3,
        )

        side_index, peak_position = np.divmod(self.peak_location, self._profile_length)

        self.force_profile = self._generate_triangular_profile(
            peak_position=int(peak_position),
            side=int(np.round(side_index)),
            peak_height=float(self.peak_magnitude),
        )

        predicted_density = self.forwarder.forward_pass(force_profile=self.force_profile)
        reward = calculate_similarity(
            reference_matrix=self.target_density,
            comparison_matrix=predicted_density,
            method="ssim",
            baseline=0.1,
            threshold=0.5,
        )

        self.last_predicted_density = predicted_density.astype(np.float32)
        density_difference = (
            self.target_density.astype(np.float32) - self.last_predicted_density
        )

        self.current_step += 1
        self.reward = reward

        # check success: is the error (observation) small everywhere?
        success = np.allclose(density_difference, 0.0, atol=0.05)

        # base termination: either out of steps or success
        terminated = success or (self.current_step >= self.max_steps)
        truncated = False

        # if success, give a big bonus on top of the normal reward
        if success:
            remaining_steps = self.max_steps - self.current_step
            reward += remaining_steps * 1.0
            logger.info(
                f"Sample {self.current_sample_index} succeeded at step {self.current_step} with reward {reward:.4f}",
            )

        info = {
            "force_profile": self.force_profile,
            "predicted_density": predicted_density,
            "reward": reward,
            "success": success,
            "current_step": self.current_step,
            "sample_index": self.current_sample_index,
        }

        observation = np.vstack(
            [density_difference, self.force_profile.astype(np.float32)],
        )  # Append action to observation
        return observation, reward, terminated, truncated, info

    def _generate_triangular_profile(
        self,
        peak_position: int,
        side: int,
        peak_height: float,
    ) -> np.ndarray:
        """Generate a 3xN force profile with one triangular peak on the selected side."""
        profile = np.zeros((3, self._profile_length), dtype=np.float32)
        peak_position = int(np.clip(peak_position, 0, self._profile_length - 1))

        # Create a triangular profile
        for j in range(self._profile_length):
            if j < peak_position:
                profile[side, j] = (peak_height / peak_position) * j
            elif j > peak_position:
                denomerator = (self._profile_length - 1 - peak_position)
                profile[side, j] = (
                    peak_height / max(denomerator, 1e-6)) * (self._profile_length - 1 - j)
            else:
                profile[side, j] = peak_height
        return profile

    def get_data_for_visualization(
        self,
    ) -> tuple[ tuple[int, np.ndarray, np.ndarray], tuple[int, np.ndarray, np.ndarray], float]:
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
