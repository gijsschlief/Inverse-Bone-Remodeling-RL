"""Training environment for reinforcement learning in bone remodeling simulation."""

import logging
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import Env, spaces
from torch.nn import Module

from bone_remodeling.src.forward_data.visualizer import plot_density_matrix
from bone_remodeling.src.rl_model.parameters import RLParameters
from bone_remodeling.src.rl_model.reward_calculation import calculate_similarity
from bone_remodeling.src.surrogate_model.loader import load_surrogate_model
from bone_remodeling.src.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.src.surrogate_model.normalizor import (
    normalize_data,
    unnormalize_data,
)
from bone_remodeling.src.surrogate_model.visualizer import plot_difference_matrix

logger = logging.getLogger(__name__)


class BoneRemodellingEnvironment(Env):
    """Gym Environment for reinforcement learning in bone remodeling simulation."""

    metadata = {"render.modes": ["human"]}  # noqa: RUF012

    def __init__(
        self,
        surrogate_model_path: Path,
        target_densities: np.ndarray,
        target_forces: np.ndarray,
        rl_parameters: RLParameters,
    ) -> None:
        """Initialize the environment with a surrogate model."""
        super().__init__()
        self.action_space: spaces.Box
        self.observation_space: spaces.Box

        # Load the surrogate model and normalization parameters
        surrogate_model_and_normalization_params = load_surrogate_model(
            surrogate_model_path,
            ReversedSurrogateModel,
        )
        assert (
            surrogate_model_and_normalization_params is not None
        ), f"Failed to load surrogate model from {surrogate_model_path}"

        (
            self.surrogate_model,
            self.x_mean,
            self.x_std,
            self.y_mean,
            self.y_std,
        ) = surrogate_model_and_normalization_params
        if self.surrogate_model is None:
            raise ValueError(
                f"Surrogate model could not be loaded from {surrogate_model_path}. Please check the file path and model type.",
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.surrogate_model.to(self.device).eval()

        self.target_densities = target_densities
        self.density_shape = target_densities[0].shape
        self._profile_length = np.max(self.density_shape)

        self.max_steps = rl_parameters.max_steps
        self.render_mode = rl_parameters.render_mode
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
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
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
        options: dict | None = None,
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
        info: dict = {options}
        return episode_observation, info

    def render(self, mode: str = "human") -> None:
        """Visualize target, current prediction, and the observation fed to the agent."""
        if mode != "human":
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")
        if self.last_predicted_density is None or self.last_predicted_density.size == 0:
            logger.warning("No density data to render.")
            return

        # On first call, create 3 grid
        if not hasattr(self, "_render_initialized"):
            self._render_fig, self._render_axes = plt.subplots(1, 3, figsize=(18, 6))
            self._render_fig.suptitle("Bone Remodeling Environment", fontsize=16)
            plt.ion()
            self._render_initialized = True
            self._last_sample_idx = None

        ax_current, ax_target, ax_obs = self._render_axes

        # Redraw target density if the sample index has changed
        if self._last_sample_idx != self.current_sample_index:
            ax_target.clear()
            plot_density_matrix(
                self.target_density,
                force_profile=self.target_force,
                title=f"Target Density (Sample {self.current_sample_index})",
                axis=ax_target,
            )
            self._last_sample_idx = self.current_sample_index

        # 1) Current / predicted density
        ax_current.clear()
        plot_density_matrix(
            self.last_predicted_density,
            force_profile=self.force_profile,
            title=f"Current Density: Step {self.current_step} / {self.max_steps}",
            axis=ax_current,
        )

        # 2) Observation (what the policy actually sees)
        ax_obs.clear()
        plot_difference_matrix(
            predicted_matrix=self.last_predicted_density,
            actual_matrix=self.target_density,
            title=f"Observation (Target - Current), Reward: {self.reward:.4f}",
            axis=ax_obs,
            color_bar=False,
        )

        self._render_fig.tight_layout()
        self._render_fig.canvas.draw()
        self._render_fig.canvas.flush_events()
        plt.pause(0.001)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Perform a step in the environment.

        Args:
        ----
            action (np.ndarray): The force profile applied to the bone.

        Returns:
        -------
            tuple: A tuple containing the observation, reward, done flag, and additional info.

        """
        peak_action, location_action = action
        self.peak_magnitude = self.peak_magnitude + np.float32(peak_action)
        self.peak_location = np.mod(
            self.peak_location + np.int8(np.round(location_action)),
            self._profile_length * 3,
        )

        side_index, peak_position = np.divmod(self.peak_location, self._profile_length)

        self.force_profile = self._generate_triangular_profile(
            peak_position=int(peak_position),
            side=int(np.round(side_index)),
            peak_height=float(self.peak_magnitude),
        )

        predicted_density = self._surrogate_model_forward()
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

        for j in range(self._profile_length):
            if j < peak_position:
                profile[side, j] = (peak_height / peak_position) * j
            elif j > peak_position:
                denom = self._profile_length - 1 - peak_position
                denom = max(denom, 1e-6)  # Prevent divide-by-zero
                profile[side, j] = (peak_height / denom) * (
                    self._profile_length - 1 - j
                )
            else:
                profile[side, j] = peak_height

        return profile

    def _surrogate_model_forward(self) -> np.ndarray:
        """Forward pass through the surrogate model.

        Args:
        ----
            force_profile (np.ndarray): The force profile applied to the bone.
            return_shape (tuple): The shape to return the predicted density.

        Returns:
        -------
            np.ndarray: The predicted density from the surrogate model.

        """
        # normalize
        if self.x_mean is not None and self.x_std is not None:
            force_profile, _, _ = normalize_data(
                self.force_profile,
                self.x_mean,
                self.x_std,
            )

        # forward pass through the surrogate model
        with torch.no_grad():
            force_profile_tensor = torch.from_numpy(
                force_profile.reshape(1, -1).astype(np.float32),
            ).to(self.device)

        assert (
            force_profile_tensor is not None
        ), "Force profile tensor is None after reshaping."
        model = cast(Module, self.surrogate_model)
        density_tensor = model(force_profile_tensor)
        assert (
            density_tensor is not None
        ), "Density tensor is None after model forward pass."

        # unnormalize
        if self.y_mean is not None and self.y_std is not None:
            density_tensor = unnormalize_data(density_tensor, self.y_mean, self.y_std)

        surrogate_density: np.ndarray = density_tensor.detach().cpu().numpy()
        return surrogate_density.reshape(self.density_shape)
