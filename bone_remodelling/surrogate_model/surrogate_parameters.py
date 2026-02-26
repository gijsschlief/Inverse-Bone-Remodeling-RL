"""Stores parameters used for training the surrogate model."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

class TrainParameters(ABC):
    """Abstract base class for training parameters."""

    model_path: Path
    device: torch.device
    epochs: int
    batch_size: int
    weight_decay: float
    alpha: float
    width: int
    depth: int
    dropout: float
    learning_rate: float
    patience_lr_scheduler: int
    factor_lr_scheduler: float
    cooldown_lr_scheduler: int
    patience: int
    min_delta: float
    log_interval: int
    log_all_for_first_epochs: int
    shuffle_data: bool
    plot_interval: int

    @abstractmethod
    def __post_init__(self) -> None:
        """Ensure that the model path exists."""
        pass


@dataclass
class SurrogateTrainParameters(TrainParameters):
    """Parameters for training the surrogate model."""

    model_path: Path
    device: torch.device
    epochs: int = 1_000

    # Hyperparameters
    batch_size: int = 64
    weight_decay: float = 1e-4
    alpha: float = 0.5
    width: int = 1024
    depth: int = 6
    dropout: float = 0.3

    # Learning rate variables
    learning_rate: float = 1e-3
    patience_lr_scheduler: int = 10
    factor_lr_scheduler: float = 0.5
    cooldown_lr_scheduler: int = 0
    patience: int = 20
    min_delta: float = 1e-4

    # Plotting variables
    log_interval: int = 10
    log_all_for_first_epochs: int = 10
    shuffle_data: bool = True
    plot_interval: int = 5

    def __post_init__(self) -> None:
        """Ensure that the model path exists."""
        if not self.model_path.parent.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
        minimal_depth: int = 4
        if self.depth < minimal_depth:
            logger.warning(f"Model depth below buildable range, using minimun of {minimal_depth}.")
        minimal_width: int = 128
        if self.width < minimal_width:
            logger.warning(f"Model width below buildable range, using minimun of {minimal_width}.")
