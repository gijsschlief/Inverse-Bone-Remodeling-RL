"""Callback designed for validation the agent so learning rate is reduced when performance flattens."""

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from bone_remodelling.rl_model.environment import BoneRemodelingEnvironment
from bone_remodelling.rl_model.metrics import MetricsContainer, ValidationSnapshot
from bone_remodelling.rl_model.parameters import (
    RLParameters,
    RunConfiguration,
)
from bone_remodelling.rl_model.reward_calculation import calculate_similarity
from bone_remodelling.rl_model.validation_environment_builder import (
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
        run_config: RunConfiguration,
        rl_parameters: RLParameters = RLParameters(),
    ) -> None:
        """Initialize the validation callback."""
        super().__init__(rl_parameters.verbose)
        self.metrics = metrics
        self.learning_rate_container = learning_rate_container
        self.validation_environment_builder = validation_environment_builder
        self.validation_forces, self.validation_densities = validation_data
        self.run_config = run_config
        self.rl_parameters = rl_parameters

        self.max_curriculum_complexity = 10
        self.current_complexity = 1
        self._reset_patience()
        self._group_validation_samples()
        self._save_agent_path()

    def _reset_patience(self) -> None:
        """Reset patience counter and best SSIM when curriculum complexity increases."""
        self.patience_counter = 0
        self.best_ssim = -np.inf

    def _group_validation_samples(self) -> None:
        """Group validation samples by complexity (e.g., number of active forces) for curriculum learning."""
        self.validation_groups: dict[int, list[int]] = {
            i: [] for i in range(1, self.max_curriculum_complexity + 1)
        }
        for i, force in enumerate(self.validation_forces):
            num_active_points = np.sum(
                np.any(np.abs(force) > self.rl_parameters.force_threshold, axis=0),
            )
            complexity = min(num_active_points, self.max_curriculum_complexity)
            if complexity == 0:
                complexity = 1
            self.validation_groups[complexity].append(i)

    def _save_agent_path(self) -> None:
        """Move the agents path to not overwrite previous agents."""
        path: Path = self.run_config.agent_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            directory = path.parent
            base_path = path.stem
            ext = path.suffix
            counter = 1
            while Path(f"{directory}/{base_path}_{counter}{ext}").exists():
                counter += 1
            path = Path(f"{directory}/{base_path}_{counter}{ext}")
        self.run_config.agent_path = path

    def _select_validation_sample(self, sample_index: int) -> int:
        """Select validation samples based on current curriculum complexity."""
        if self.validation_groups[self.current_complexity]:
            return self.validation_groups[self.current_complexity][
                sample_index % len(self.validation_groups[self.current_complexity])
            ]
        return sample_index % len(self.validation_forces)

    def _on_step(self) -> bool:
        """Perform validation at specified intervals and adjust learning rate if performance plateaus."""
        if self.num_timesteps % self.run_config.validation_frequency != 0:
            return True

        self.current_complexity = self.training_env.get_attr("current_max_complexity")[
            0
        ]
        mean_ssim = self._ssim_calculation()
        logger.info(f"[Val @ {self.num_timesteps}] mean final SSIM = {mean_ssim:.4f}")

        snapshot = ValidationSnapshot(
            step=self.num_timesteps,
            ssim=mean_ssim,
            complexity=self.current_complexity,
            learning_rate=self.learning_rate_container["value"],
            patience=self.patience_counter,
        )
        self.metrics.add_snapshot(snapshot)

        if (
            mean_ssim >= self.rl_parameters.curriculum_threshold
            and self.current_complexity <= self.max_curriculum_complexity
        ):
            return self._next_curriculum_level()

        return self._detect_plateau(mean_ssim)

    def _ssim_calculation(self) -> float:
        """Calculate mean SSIM over a set of validation samples using the same evaluation procedure as in evaluate_agent."""
        ssim_scores = []
        for sample_counter in range(self.run_config.validation_size):
            index = self._select_validation_sample(sample_counter)
            force = self.validation_forces[index]
            target = self.validation_densities[index]

            validation_environment: BoneRemodelingEnvironment = (
                self.validation_environment_builder(force, target)
            )
            observation, _ = validation_environment.reset()
            done = False
            while not done:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = validation_environment.step(
                    action,
                )
                done = terminated or truncated

            sample_info, estimate_info, _ = (
                validation_environment.get_data_for_visualization()
            )

            score = calculate_similarity(
                reference_matrix=sample_info[2],
                comparison_matrix=estimate_info[2],
                method="ssim",
            )
            ssim_scores.append(score)
        return float(np.mean(ssim_scores))

    def _save_checkpoint(self) -> None:
        """Save the agent and training stats at the current checkpoint."""
        stats_to_save = {
            "current_learning_rate": self.learning_rate_container["value"],
            "current_patience": self.patience_counter,
            "current_step": self.num_timesteps,
            "max_ssim": self.best_ssim,
            "current_complexity": self.current_complexity,
            "complexity_jumps": self.metrics.complexity_jumps,
            "history": [asdict(s) for s in self.metrics.history],
            "episode_rewards": self.metrics.episode_rewards,
            "episode_end_timesteps": self.metrics.episode_end_timesteps,
        }

        # Intentionally adding a dynamic attribute
        self.model.custom_stats = stats_to_save  # pyright: ignore[reportAttributeAccessIssue]
        self.model.save(self.run_config.agent_path, include=["custom_stats"])

    def _detect_plateau(self, last_ssim: float) -> bool:
        """Detect if validation performance has plateaued and decide whether to reduce learning rate."""
        if last_ssim > self.best_ssim:
            self.best_ssim = last_ssim
            self.patience_counter = 0
            logger.info(f"[New best SSIM: {self.best_ssim:.4f}")
            self._save_checkpoint()
            return True

        if self.patience_counter < self.rl_parameters.patience:
            self.patience_counter += 1
            logger.warning(
                f"SSIM did not improve, patience counter: {self.patience_counter}, best SSIM: {self.best_ssim:.4f}",
            )
            self._save_checkpoint()
            return True

        if self.patience_counter >= self.rl_parameters.patience:
            logger.warning(
                f"[SSIM did not improve, best SSIM: {self.best_ssim:.4f}] Validation plateau detected, reducing learning rate.",
            )
            self._reset_patience()
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
        self._save_checkpoint()
        return True

    def _next_curriculum_level(self) -> bool:
        """Advance to the next curriculum complexity level if mastery is achieved."""
        logger.info(
            f"Mastery of Level {self.current_complexity} achieved! Advancing...",
        )
        self.metrics.complexity_jumps.append(
            (self.num_timesteps, self.current_complexity + 1),
        )
        self.training_env.env_method("increase_curriculum_complexity")
        self._reset_patience()
        self.learning_rate_container["value"] = self.rl_parameters.learning_rate
        self._save_checkpoint()
        return True

    def _on_training_end(self) -> None:
        logger.info("Training finished. Saving final checkpoint...")
        self._save_checkpoint()
