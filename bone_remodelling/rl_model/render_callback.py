"""Callback to render the environment at specified intervals."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from bone_remodelling.forward_data.force_profile_generator import ForceProfileGenerator
from bone_remodelling.forward_model.density_visualizer import plot_density_matrix
from bone_remodelling.rl_model.parameters import RLParameters
from bone_remodelling.surrogate_model.visualizer import plot_difference_matrix

logger = logging.getLogger(__name__)


class RenderCallback(BaseCallback):
    """Callback to render the environment at specified intervals."""

    def __init__(
        self,
        force_generator: ForceProfileGenerator,
        render_freq: int = 10,
        environment_index: int = 0,
        rl_parameters: RLParameters = RLParameters(),
    ) -> None:
        """Initialize the render callback.

        Args:
        ----
            force_generator (ForceProfileGenerator): Force profile generator to get the force mask for visualization.
            render_freq (int): Frequency of rendering in terms of steps.
            environment_index (int): Index of the environment to render.
            rl_parameters (RLParameters): RL parameters for the callback.

        """
        super().__init__(rl_parameters.verbose)
        self.render_freq = render_freq
        self.environment_index = environment_index
        self.max_steps = rl_parameters.max_steps
        self.force_mask = force_generator.generate_force_mask()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.render_freq == 0:
            get_data = self.training_env.get_attr("get_data_for_visualization")[
                self.environment_index
            ]
            sample_information, estimate_information, reward = get_data()
            self.render(
                sample_information,
                estimate_information,
                reward,
            )
            return True
        return True

    def render(
        self,
        sample_information: tuple[int, np.ndarray, np.ndarray],
        estimate_information: tuple[int, np.ndarray, np.ndarray],
        reward: float,
    ) -> None:
        """Visualize target, current prediction, and the observation fed to the agent."""
        current_sample_index, target_force, target_density = sample_information
        current_step, force_profile, last_predicted_density = estimate_information

        # On first call, create 3 grid
        if not hasattr(self, "_render_initialized"):
            self._render_fig, self._render_axes = plt.subplots(1, 3, figsize=(18, 6))
            plt.ion()
            self._render_initialized = True
            self._last_sample_idx = None

        complexity = self.training_env.get_attr("current_max_complexity")[0]
        self._render_fig.suptitle(
            f"Bone Remodeling Environment | Level: {complexity} | Global Step: {self.num_timesteps}",
            fontsize=16,
        )
        ax_current, ax_target, ax_obs = self._render_axes

        # Redraw target density if the sample index has changed
        if self._last_sample_idx != current_sample_index:
            ax_target.clear()
            plot_density_matrix(
                target_density,
                force_data=(target_force, self.force_mask),
                title=f"Target Density (Sample {current_sample_index})",
                axis=ax_target,
            )
            self._last_sample_idx = current_sample_index

        # 1) Current / predicted density
        ax_current.clear()
        plot_density_matrix(
            last_predicted_density,
            force_data=(force_profile, self.force_mask),
            title=f"Current Density: Step {current_step} / {self.max_steps}",
            axis=ax_current,
        )

        # 2) Observation (what the policy actually sees)
        ax_obs.clear()
        plot_difference_matrix(
            predicted_matrix=last_predicted_density,
            actual_matrix=target_density,
            title=f"Observation (Target - Current), Reward: {reward:.4f}",
            axis=ax_obs,
        )

        self._render_fig.tight_layout()
        self._render_fig.canvas.draw()
        self._render_fig.canvas.flush_events()
        plt.pause(0.001)
