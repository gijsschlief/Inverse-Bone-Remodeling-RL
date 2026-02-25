"""Trainer used to train surrogate model."""

import logging
import time
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import optuna
import torch

from bone_remodelling.surrogate_model.loader import SurrogatePredictor
from bone_remodelling.surrogate_model.optimiser_scheduler_dataloader import (
    build_dataloader,
    build_learning_rate_scheduler,
    build_optimizer,
)
from bone_remodelling.surrogate_model.surrogate_parameters import (
    TrainParameters,
)
from bone_remodelling.surrogate_model.visualizer import plot_loss

logger = logging.getLogger(__name__)


class SurrogateModelTrainer:
    """Trainer class for the surrogate model."""

    def __init__(
        self,
        predicter: SurrogatePredictor,
        train_parameters: TrainParameters,
        loss_function: Callable,
    ) -> None:
        """Initialize the SurrogateModelTrainer."""
        self.predicter = predicter
        self.train_parameters = train_parameters

        self.predicter.model.to(self.train_parameters.device)
        self.loss_function = loss_function
        self.optimizer = build_optimizer(self.predicter.model, self.train_parameters)
        self.scheduler = build_learning_rate_scheduler(
            self.optimizer,
            self.train_parameters,
        )

    def time_training(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Time the training process of the surrogate model."""
        start_time = time.time()
        self.train(train_data, val_data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds.")
        return elapsed_time

    def train(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
        trial: optuna.trial.Trial | None = None,
    ) -> None:
        """Train the surrogate model."""
        x_train_np = self.predicter.set_input_normalize(train_data[0])
        x_val_np = self.predicter.normalize_input(val_data[0])
        y_train_np = self.predicter.set_output_normalize(train_data[1])
        y_val_np = self.predicter.normalize_output(val_data[1])

        self.dataloader = build_dataloader(
            torch.tensor(x_train_np, dtype=torch.float32),
            torch.tensor(y_train_np, dtype=torch.float32),
            batch_size=self.train_parameters.batch_size,
            shuffle=self.train_parameters.shuffle_data,
        )

        x_val: torch.Tensor = torch.tensor(x_val_np, dtype=torch.float32).to(
            self.train_parameters.device,
        )
        y_val: torch.Tensor = torch.tensor(y_val_np, dtype=torch.float32).to(
            self.train_parameters.device,
        )

        self.best_val_loss = float("inf")

        self.training_losses: list[float] = []
        self.validation_losses: list[float] = []
        self.learning_rates: list[float] = []

        epoch, epochs_no_improve = 0, 0
        self.best_model_state: dict[str, torch.Tensor] | None = None
        batch_count = len(self.dataloader)
        self.best_model_state = deepcopy(self.predicter.model.state_dict())

        for epoch in range(self.train_parameters.epochs):
            self.predicter.model.train()
            total_loss = 0.0
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)

            for batch_x, batch_y in self.dataloader:
                batch_x_device = batch_x.to(self.train_parameters.device)
                batch_y_device = batch_y.to(self.train_parameters.device)

                self.optimizer.zero_grad()
                logits = self.predicter.model(batch_x_device)
                loss = self.loss_function(logits, batch_y_device)
                self.warn_unexpected_loss(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            average_training_loss = total_loss / batch_count
            self.training_losses.append(average_training_loss)
            self.predicter.model.eval()
            with torch.no_grad():
                val_logits = self.predicter.model(x_val)
                validation_loss = self.loss_function(val_logits, y_val)

            self.validation_losses.append(validation_loss.item())

            # Communication with optuna
            if trial is not None:
                trial.report(validation_loss, epoch)
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at epoch {epoch}.")
                    raise optuna.exceptions.TrialPruned

            self.scheduler.step(validation_loss)

            early_stop, epochs_no_improve = self.early_stopping_check(
                self.validation_losses[-1],
                epochs_no_improve,
            )
            self.log_training_progress(epoch)
            if (epoch + 1) % self.train_parameters.plot_interval == 0:
                plot_loss(
                    self.training_losses,
                    self.validation_losses,
                    self.learning_rates,
                )
            if early_stop:
                logger.info(
                    f"Early stopping at epoch {epoch} (no improvement in {self.train_parameters.patience} epochs).",
                )
                break
        self.reload_and_save_best_model()
        plot_loss(self.training_losses, self.validation_losses, self.learning_rates)

    def warn_unexpected_loss(self, loss: torch.Tensor) -> None:
        """Check if the loss is a good value (not NaN, Inf, or non-finite)."""
        if torch.isnan(loss):
            logger.error("Loss is NaN, skipping this batch.")
        elif torch.isinf(loss):
            logger.error("Loss is Inf, skipping this batch.")
        elif not torch.isfinite(loss):
            logger.error("Loss is not finite, skipping this batch.")
        else:
            return

    def early_stopping_check(
        self,
        current_val_loss: float,
        epochs_no_improve: int,
    ) -> tuple[bool, int]:
        """Check if early stopping criteria are met."""
        if current_val_loss + self.train_parameters.min_delta < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_model_state = deepcopy(self.predicter.model.state_dict())
            epochs_no_improve = 0
            logger.info(
                f"Validation loss improved to {self.best_val_loss:.4f}. Saving model state.",
            )
            return False, epochs_no_improve
        epochs_no_improve += 1
        if epochs_no_improve >= self.train_parameters.patience:
            return True, epochs_no_improve
        return False, epochs_no_improve

    def log_training_progress(self, epoch: int) -> None:
        """Log the training progress at specified intervals."""
        if (
            epoch < self.train_parameters.log_all_for_first_epochs
            or (epoch + 1) % self.train_parameters.log_interval == 0
        ):
            logger.info(
                f"Epoch {epoch + 1}/{self.train_parameters.epochs}, Train Loss: {self.training_losses[-1]:.4f}, Validation Loss: {self.validation_losses[-1]:.4f}",
            )

    def reload_and_save_best_model(self) -> None:
        """Reload the best model state after training."""
        if self.best_model_state is not None:
            self.predicter.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model state after training.")
            history = {
                "training_losses": self.training_losses,
                "validation_losses": self.validation_losses,
                "learning_rates": self.learning_rates,
            }
            self.predicter.save_model_with_versioning(
                self.train_parameters.model_path,
                history=history,
            )
