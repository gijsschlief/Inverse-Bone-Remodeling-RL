"""Callback designed for validation the agent so learning rate is reduced when performance flattens."""

import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from bone_remodeling.src.rl_model.forward_pass import (
    ForwardPass,
)
from bone_remodeling.src.rl_model.metrics import MetricsContainer
from bone_remodeling.src.rl_model.parameters import RLParameters
from bone_remodeling.src.rl_model.reward_calculation import calculate_similarity
from bone_remodeling.src.rl_model.validation_environment_builder import (
    ValidationEnvironmentBuilder,
)

logger = logging.getLogger(__name__)


class ValidationCallback(BaseCallback):
    """Validate agent based on 10 forward passes with the final one using the fenics model."""

    def __init__(
        self,
        metrics: MetricsContainer,
        learning_rate_container: dict[str, float],
        validation_data: tuple[np.ndarray, np.ndarray],
        validation_environment_builder: ValidationEnvironmentBuilder,
        final_forwarder: ForwardPass,
        validation_frequency: int = 100_000,
        rl_parameters: RLParameters = RLParameters(),
    ) -> None:
        """Initialize the validation callback."""
        super().__init__(rl_parameters.verbose)
        self.metrics = metrics
        self.metrics.validation_steps.append(0)
        self.metrics.validation_ssim.append(0.0)
        self.learning_rate_container = learning_rate_container
        self.validation_environment_builder = validation_environment_builder
        self.fenics_forwarder = final_forwarder
        self.validation_forces, self.validation_densities = validation_data
        self.validation_frequency = validation_frequency
        self.rl_parameters = rl_parameters

        self.best_ssim = -np.inf
        self.patience_counter = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.validation_frequency != 0:
            return True

        mean_ssim = self._ssim_calculation()
        logger.info(f"[Val @ {self.num_timesteps}] mean final SSIM = {mean_ssim:.4f}")

        self.metrics.validation_steps.append(self.num_timesteps)
        self.metrics.validation_ssim.append(mean_ssim)
        return self._detect_plateau(mean_ssim)

    def _ssim_calculation(self) -> float:
        ssim_scores = []
        for force, target in zip(self.validation_forces, self.validation_densities):
            # 1) surrogate rollout
            validation_environment = self.validation_environment_builder(force, target)
            observation, _ = validation_environment.reset()
            for _ in range(self.rl_parameters.max_steps - 1):
                action, _ = self.model.predict(observation, deterministic=True)
                observation, _, done, _, _ = validation_environment.step(action)
                if done:
                    break

            # 2) true FEniCS solve of the final force profile
            true_density = self.fenics_forwarder.forward_pass(
                validation_environment.force_profile,
            )

            # 3) compute SSIM vs target
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
        if last_ssim > self.best_ssim:
            self.best_ssim = last_ssim
            self.patience_counter = 0
            logger.info(f"[New best SSIM: {self.best_ssim:.4f}")
            return True

        # If the SSIM is not improving, increase patience counter, reduce learning rate or stop training
        # increase patience counter
        if self.patience_counter < self.rl_parameters.patience:
            self.patience_counter += 1
            logger.warning(
                f"SSIM did not improve, patience counter: {self.patience_counter}, best SSIM: {self.best_ssim:.4f}",
            )
            return True

        # reduce learning rate if patience is exceeded
        if self.patience_counter >= self.rl_parameters.patience:
            logger.warning(
                f"[SSIM did not improve, best SSIM: {self.best_ssim:.4f}] Validation plateau detected, reducing learning rate.",
            )
            self.best_ssim = -np.inf
            self.patience_counter = 0
            return self._learning_rate_reducer()
        return False

    def _learning_rate_reducer(self) -> bool:
        """Reduce learning rate if plateau is detected."""
        old_learning_rate = self.learning_rate_container["value"]
        new_learning_rate = old_learning_rate * self.rl_parameters.learning_rate_decay

        if new_learning_rate <= self.rl_parameters.minimum_learning_rate:
            logger.warning("LR is already at minimum, RL simulation has converged!")
            return False

        self.learning_rate_container["value"] = new_learning_rate

        logger.warning(
            f"Reducing LR from {old_learning_rate:.2e} to {new_learning_rate:.2e}",
        )
        return True
