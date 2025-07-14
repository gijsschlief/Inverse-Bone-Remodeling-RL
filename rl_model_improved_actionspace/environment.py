"""Training environment for reinforcement learning in bone remodeling simulation."""

import logging
import os
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from bone_remodeling.forward_model.data_reader import forward_data_reader
from bone_remodeling.rl_model.reward_calculation import calculate_similarity
from bone_remodeling.surrogate_model.loader import load_surrogate_model
from bone_remodeling.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.surrogate_model.normalizor import (
    normalize_data,
    unnormalize_data,
)
from bone_remodeling.surrogate_model.splitter import splitting
from bone_remodeling.surrogate_model.visualizer import (
    plot_density_matrix,
    plot_difference_matrix,
)
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import Module


class BoneRemodellingEnvironment(Env):
    """Gym Environment for reinforcement learning in bone remodeling simulation."""

    metadata = {"render.modes": ["human"]}  # noqa: RUF012

    def __init__(
        self,
        surrogate_model_path: Path,
        target_densities: np.ndarray,
        target_forces: np.ndarray,
        max_steps: int = 50,
        force_boundary: float = 30,
        density_constraint: float = 1.73,
        render_mode: str = "human",
    ) -> None:
        """Initialize the environment with a surrogate model."""
        super().__init__()
        self.action_space: spaces.Box
        self.observation_space: spaces.Box

        # Load the surrogate model and normalization parameters
        surrogate_model_and_normalization_params = load_surrogate_model(
            surrogate_model_path, ReversedSurrogateModel
        )
        assert (
            surrogate_model_and_normalization_params is not None
        ), f"Failed to load surrogate model from {surrogate_model_path}"

        self.surrogate_model, self.x_mean, self.x_std, self.y_mean, self.y_std = (
            surrogate_model_and_normalization_params
        )
        if self.surrogate_model is None:
            raise ValueError(
                f"Surrogate model could not be loaded from {surrogate_model_path}. Please check the file path and model type."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.surrogate_model.to(self.device).eval()

        self.target_densities = target_densities
        self.return_shape = target_densities[0].shape
        self._profile_length = np.max(self.return_shape)
        self.max_steps = max_steps
        self.force_profile = np.zeros(
            (3, self._profile_length), dtype=np.float32
        )  # Default force profile
        self.render_mode = render_mode
        self.target_forces = target_forces
        self.num_samples = len(target_densities)
        self.current_sample_index = 0

        # Define the action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, -force_boundary], dtype=np.float32),
            high=np.array(
                [self._profile_length - 1, 1, 1, force_boundary], dtype=np.float32
            ),
            dtype=np.float32,
        )

        # Observation space of the agent
        self.grid_size = int(np.prod(self.return_shape))
        observation_space_lower_bounds = np.concatenate(
            [
                np.full(self.grid_size, -density_constraint, dtype=np.float32),
                np.array(
                    [
                        self.action_space.low[0],
                        self.action_space.low[1],
                        self.action_space.low[2],
                        self.action_space.low[3],
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        observation_space_upper_bounds = np.concatenate(
            [
                np.full(self.grid_size, +density_constraint, dtype=np.float32),
                np.array(
                    [
                        self.action_space.high[0],
                        self.action_space.high[1],
                        self.action_space.high[2],
                        self.action_space.high[3],
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        self.observation_space = spaces.Box(
            low=observation_space_lower_bounds,
            high=observation_space_upper_bounds,
            dtype=np.float32,
        )

        # Episode variables
        self.current_step = 0
        self.last_predicted_density = np.zeros(self.return_shape, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
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

        # Pick a new random sample
        self.current_sample_index = np.random.randint(self.num_samples)

        # Update target for this episode
        self.target_density = self.target_densities[self.current_sample_index]
        self.target_force = self.target_forces[self.current_sample_index]
        self.return_shape = self.target_density.shape
        self.last_predicted_density = np.zeros(self.return_shape, dtype=np.float32)
        difference = (
            self.target_density.astype(np.float32) - self.last_predicted_density
        ).ravel()
        episode_observation = np.concatenate(
            [difference, np.zeros(4, dtype=np.float32)], axis=0
        )
        info: dict = {}
        return episode_observation, info

    def render(self, mode: str = "human") -> None:
        """Visualize target, current prediction, and the observation fed to the agent."""
        if mode != "human":
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")
        if self.last_predicted_density is None or self.last_predicted_density.size == 0:
            logging.warning("No density data to render.")
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
        peak_position, top_index, side_index, peak_height = action
        peak_position = int(
            np.clip(
                round(peak_position),
                self.action_space.low[0],
                self.action_space.high[0],
            )
        )
        top_index = int(
            np.clip(
                round(top_index), self.action_space.low[1], self.action_space.high[1]
            )
        )
        side_index = int(
            np.clip(
                round(side_index), self.action_space.low[2], self.action_space.high[2]
            )
        )  # Ensure side_index is within [0, 2]
        peak_height = float(
            np.clip(peak_height, self.action_space.low[3], self.action_space.high[3])
        )

        # Generate force profile
        side_index = (
            (2 if side_index == 0 else 1) if top_index == 0 else 0
        )  # Left/right if top_index==0, else top side

        self.force_profile = self._generate_triangular_profile(
            peak_position=peak_position, side=side_index, peak_height=peak_height
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
        observation = (
            self.target_density.astype(np.float32) - self.last_predicted_density
        )

        self.current_step += 1
        self.reward = reward

        # check success: is the error (observation) small everywhere?
        success = np.allclose(observation, 0.0, atol=0.05)

        # base termination: either out of steps or success
        terminated = success or (self.current_step >= self.max_steps)
        truncated = False

        # if success, give a big bonus on top of the normal reward
        if success:
            remaining_steps = self.max_steps - self.current_step
            reward += remaining_steps * 1.0
            logging.info(
                f"Sample {self.current_sample_index} succeeded at step {self.current_step} with reward {reward:.4f}"
            )

        info = {
            "force_profile": self.force_profile,
            "predicted_density": predicted_density,
            "reward": reward,
            "success": success,
            "current_step": self.current_step,
            "sample_index": self.current_sample_index,
        }

        observation = np.concatenate(
            ([observation.ravel(), action]), axis=0
        )  # Append action to observation
        return observation, reward, terminated, truncated, info

    def _generate_triangular_profile(
        self, peak_position: int, side: int, peak_height: float
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
            force_profile, _, _, _, _ = normalize_data(
                self.force_profile, None, None, self.x_mean, self.x_std
            )

        # forward pass through the surrogate model
        with torch.no_grad():
            force_profile_tensor = torch.from_numpy(
                force_profile.reshape(1, -1).astype(np.float32)
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
        return surrogate_density.reshape(self.return_shape)


class RenderCallback(BaseCallback):
    """Callback to render the environment at specified intervals."""

    def __init__(self, render_freq: int = 10, verbose: int = 0) -> None:
        """Initialize the render callback.

        Args:
        ----
            render_freq (int): Frequency of rendering in terms of steps.
            verbose (int): Verbosity level.

        """
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        vecenv = cast(DummyVecEnv, self.training_env)
        if self.n_calls % self.render_freq == 0:
            vecenv.envs[0].render()
        return True


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
        plt.ylim(-1, 10)
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.title("Episode Reward over Training")
        plt.tight_layout()
        plt.savefig(self.out_path)
        plt.close()
        if self.verbose:
            logging.info(f"Saved reward plot to {self.out_path}")


def save_model_safely(model: PPO, path: Path) -> Path:
    """Save the model to a file, ensuring no overwriting of existing files."""
    if os.path.exists(path):
        base_path, ext = os.path.splitext(path)
        counter = 1
        while os.path.exists(f"{base_path}_{counter}{ext}"):
            counter += 1
        path = Path(f"{base_path}_{counter}{ext}")
    model.save(path)
    return Path(path)


def find_latest_agent(path: Path) -> Path:
    """Find the latest agent file in the specified directory."""
    if os.path.exists(path):
        base_path, ext = os.path.splitext(path)
        counter = 1
    while os.path.exists(f"{base_path}_{counter}{ext}"):
        counter += 1
    return Path(f"{base_path}_{counter-1}{ext}")


def main(agent_path: Path, data_path: Path, surrogate_path: Path) -> None:
    """Designs and trains a reinforcement learning agent for bone remodeling.

    This function initializes the bone remodeling environment, loads the surrogate model,
    and trains a reinforcement learning agent using the Proximal Policy Optimization (PPO)
    algorithm. It reads training data, splits it into training sets, and uses callbacks for
    rendering and saving rewards during training.

    Args:
    ----
        agent_path (Path): Path to the agent model file.
        data_path (Path): Path to the data file for training.
        surrogate_path (Path): Path to the surrogate model file.

    """
    result = forward_data_reader(data_path)
    assert result is not None, "Failed to read data from the specified path."
    _, target_forces, target_densities = result

    train_densities, _, _, train_forces, _, _ = splitting(
        target_densities, target_forces, random_state=0
    )
    logging.info(f"Training RL agent on {len(train_densities)} samples.")

    remodeling_environment = BoneRemodellingEnvironment(
        surrogate_model_path=surrogate_path,
        target_densities=train_densities,
        target_forces=train_forces,
        max_steps=100,
    )

    logging.info("Environment created,loading agent if it exists.")
    if agent_path is None:
        model = PPO("MlpPolicy", remodeling_environment, verbose=1)
    else:
        latest_agent_path = find_latest_agent(agent_path)
        if latest_agent_path is None:
            model = PPO("MlpPolicy", remodeling_environment, verbose=1)
        else:
            model = PPO.load(latest_agent_path, env=remodeling_environment)
            model.set_env(remodeling_environment)

    logging.info(
        f"Starting training with agent at {latest_agent_path if latest_agent_path else 'new model'}."
    )

    model.learn(
        total_timesteps=1000,
        callback=[
            RenderCallback(render_freq=1),
            RewardSavingCallback(
                out_path="/home/gijs/Desktop/Thesis/data/figures/reward_curve_RL_special.png"
            ),
        ],
    )
    logging.info("Training complete.")

    saved_path = save_model_safely(model, agent_path)
    logging.info(f"Model saved to {saved_path}")


if __name__ == "__main__":
    AGENT_PATH = Path("/home/gijs/Desktop/Thesis/data/agents/trained_agent_special.zip")
    DATA_PATH = Path(
        "/home/gijs/Desktop/Thesis/data/raw/training_triangular_profiles_10000_samples_0708_1959.json"
    )
    SURROGATE_PATH = Path("/home/gijs/Desktop/Thesis/data/models/trained_model_1.pth")
    main(agent_path=AGENT_PATH, data_path=DATA_PATH, surrogate_path=SURROGATE_PATH)
