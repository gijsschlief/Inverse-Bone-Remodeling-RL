"""Training environment for reinforcement learning in bone remodeling simulation."""

import logging
from pathlib import Path

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
from gymnasium import Env, spaces
from stable_baselines3.common.callbacks import BaseCallback

# TODO: Pretrain using supervised learning on the surrogate model
# TODO: Learn on all samples in the dataset
# TODO: Load previous models

class BoneRemodellingEnvironment(Env):
    """Gym Environment for reinforcement learning in bone remodeling simulation."""

    metadata = {"render.modes": ["human"]}  # noqa: RUF012

    def __init__(self, model_path: str, target_density: np.ndarray, target_forces: np.ndarray, max_steps: int = 50, force_boundary: float = 30, density_constraint: float = 1.73, render_mode: str = "human") -> None:
        """Initialize the environment with a surrogate model."""
        super().__init__()

        # Load the surrogate model and normalization parameters
        surrogate_model_and_normalization_params = load_surrogate_model(model_path, ReversedSurrogateModel)
        if surrogate_model_and_normalization_params is None:
            raise ValueError("Failed to load the surrogate model.")
        self.surrogate_model, self.x_mean, self.x_std, self.y_mean, self.y_std = surrogate_model_and_normalization_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.surrogate_model.to(self.device).eval()

        self.target_density = target_density
        self.return_shape = target_density.shape
        self.max_steps = max_steps
        self.force_profile = np.zeros((3, np.max(self.return_shape)), dtype=np.float32)  # Default force profile
        self.render_mode = render_mode
        self.target_forces = target_forces

        # Define the action and observation spaces
        self.action_space = spaces.Box(
            low=-force_boundary,
            high=force_boundary,
            shape=(3, 10),
            dtype=np.float32,
        )

        # Observation space of the agent
        self.observation_space = spaces.Box(
            low=0.0,
            high=density_constraint,
            shape=self.return_shape,
            dtype=np.float32,
        )

        # Episode variables
        self.current_step = 0
        self.last_density = np.zeros(self.return_shape, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
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
        episode_observation = self.last_density
        info: dict = {}
        return episode_observation, info

    def render(self, mode: str ="human") -> None:
        """Visualize the current density as a heatmap."""
        import matplotlib.pyplot as plt

        from surrogate_model.visualizer import plot_density_matrix
        if mode != "human":
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")
        if self.last_density is None or self.last_density.size == 0:
            logging.warning("No density data to render.")
            return

    # Initialize figure and axes only once
        if not hasattr(self, "_render_initialized"):
            self._render_fig, self._render_axes = plt.subplots(1, 2, figsize=(14, 7))
            self._render_fig.suptitle("Bone Remodeling Environment", fontsize=16)
            plt.ion()
            self._render_initialized = True

            # Plot target density ONCE (doesn't change)
            ax_current, ax_target = self._render_axes
            ax_target.clear()
            target_force_profile = getattr(self, "target_forces", None)
            plot_density_matrix(
                self.target_density,
                force_profile=target_force_profile,
                title="Target Density",
                axis=ax_target,
            )

        # Always update current density
        ax_current, _ = self._render_axes
        ax_current.clear()
        plot_density_matrix(
            self.last_density,
            force_profile=self.force_profile,
            title="Current Density",
            axis=ax_current,
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
        self.force_profile = np.clip(action, self.action_space.low, self.action_space.high)

        predicted_density = self._surrogate_model_forward()
        reward = calculate_similarity(
            reference_matrix=self.target_density,
            comparison_matrix=predicted_density,
            method="mse",
            baseline=0.1,
            threshold=0.5
            )

        observation = self.target_density.astype(np.float32)
        self.last_density = predicted_density.astype(np.float32)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "force_profile": self.force_profile,
            "predicted_density": predicted_density,
            "reward": reward,
            "current_step": self.current_step
        }
        return observation, reward, terminated, truncated, info

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
        if self.x_mean is not None:
            force_profile, _, _, _, _ = normalize_data(self.force_profile, None, None, self.x_mean, self.x_std)

        # forward pass through the surrogate model
        with torch.no_grad():
            force_profile_tensor = torch.from_numpy(force_profile.reshape(1, -1).astype(np.float32)).to(self.device)

        density_tensor: torch.Tensor = self.surrogate_model(force_profile_tensor)
        surrogate_density = density_tensor.detach().cpu().numpy()

        # unnormalize
        if self.y_mean is not None:
            surrogate_density = unnormalize_data(surrogate_density, self.y_mean, self.y_std)
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
        if self.n_calls % self.render_freq == 0:
            self.training_env.envs[0].render()
        return True

def main() -> None:
    """Demonstrates the environment and reward calculation."""
    import stable_baselines3 as sb3

    directory_path = Path("/home/gijs/Desktop/Thesis/data/raw/")
    result = forward_data_reader(directory_path)
    if result is not None:
        _, target_forces, target_densities = result

    remodeling_environment = BoneRemodellingEnvironment(
        model_path="/home/gijs/Desktop/Thesis/data/models/trained_model_1.pth",
        target_density=target_densities[0],  # Use the first target density for demonstration
        target_forces=target_forces[0],
        max_steps=100,
    )

    model = sb3.PPO("MlpPolicy", remodeling_environment, verbose=1)

    pretrained_path = "/home/gijs/Desktop/Thesis/data/pretrained_agents/trained_agent_weights.pth"
    try:
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
        model.policy.load_state_dict(pretrained_dict)
        logging.info("Loaded pretrained weights into PPO agent.")
    except Exception as e:
        logging.warning(f"Could not load pretrained policy: {e}")

    model.learn(total_timesteps=50_000, callback=RenderCallback(render_freq=1000))
    logging.info("Training complete.")

    model.save("/home/gijs/Desktop/Thesis/data/agents/trained_agent.zip")

if __name__ == "__main__":
    main()
