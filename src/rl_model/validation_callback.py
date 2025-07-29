"""Callback designed for validation the agent so learning rate is reduced when performance flattens."""

import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from bone_remodeling.src.rl_model.forward_pass import (
    ForwardPass,
)
from bone_remodeling.src.rl_model.parameters import RLParameters
from bone_remodeling.src.rl_model.reward_calculation import calculate_similarity
from bone_remodeling.src.rl_model.validation_environment_builder import (
    ValidationEnvironmentBuilder,
)

logger = logging.getLogger(__name__)

class ValidationCallback(BaseCallback):
    """Validate agent based on 10 forward passes with the final one using the fenics model."""

    def __init__(self,
                validation_data: tuple[np.ndarray, np.ndarray],
                validation_environment_builder: ValidationEnvironmentBuilder,
                final_forwarder: ForwardPass,
                validation_frequency: int = 100_000,
                rl_parameters: RLParameters = RLParameters()) -> None:
        """Initialize the validation callback."""
        super().__init__(rl_parameters.verbose)
        self.validation_environment_builder = validation_environment_builder
        self.fenics_forwarder = final_forwarder
        self.validation_forces, self.validation_densities = validation_data
        self.validation_frequency = validation_frequency
        self.rl_parameters = rl_parameters
        self.best_ssim = 0.0
        self.patience_counter = 0


    def _on_step(self) -> bool:
        if self.num_timesteps % self.validation_frequency != 0:
            return True

        mean_ssim = self._ssim_calculation()

        if self._detect_plateau(mean_ssim):
            self._learning_rate_reducer()

        logger.info(f"[Val @ {self.num_timesteps}] mean final SSIM = {mean_ssim:.4f}")
        return True

    def _ssim_calculation(self) -> float:
        ssim_scores = []
        for force, target in zip(self.validation_forces, self.validation_densities):
            # 1) surrogate rollout
            validation_environment = self.validation_environment_builder(force, target)
            observation, _ = validation_environment.reset()
            for _ in range(self.rl_parameters.max_steps - 1):
                action, _ = self.model.predict(observation, deterministic=True)
                observation, _, done, _ = validation_environment.step(action)
                if done:
                    break

            # 2) final action
            final_action, _ = self.model.predict(observation, deterministic=True)

            # 3) true FEniCS solve of that final action
            true_density = self.fenics_forwarder.forward_pass(final_action)

            # 4) compute SSIM vs target
            score = calculate_similarity(
                reference_matrix=target,
                comparison_matrix=true_density,
                method="ssim",
                baseline=0.1,
                threshold=0.5,
            )
            ssim_scores.append(score)
        return float(np.mean(ssim_scores))

    def _detect_plateau(self, last_ssim: float) -> bool:
        if self.n_calls < self.validation_frequency * 2:
            return False

        if last_ssim < self.best_ssim * self.rl_parameters.patience_threshold:
            self.patience_counter += 1
            if self.patience_counter >= self.rl_parameters.patience:
                logger.warning(f"[Val @ {self.num_timesteps}] Validation plateau detected, reducing learning rate.")
                self.best_ssim = last_ssim
                self.patience_counter = 0
                return True
        else:
            self.patience_counter = 0

        return False

    def _learning_rate_reducer(self) -> None:
        """Reduce learning rate if plateau is detected."""
        self.patience_counter = 0
        new_lr = self.model.learning_rate * self.rl_parameters.learning_rate_decay
        logger.info(f"Reducing learning rate from {self.model.learning_rate} to {new_lr}.")
        self.model.learning_rate = new_lr
        self.model.optimizer.param_groups[0]['lr'] = new_lr
        self.best_ssim = 0.0
