"""Stores parameters used for training the surrogate model."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class SurrogateTrainParameters:
    """Parameters for training the surrogate model."""

    model_path: Path
    device: torch.device
    epochs: int = 1_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience_lr_scheduler: int = 25
    factor_lr_scheduler: float = 0.5
    cooldown_lr_scheduler: int = 10
    patience: int = 50
    min_delta: float = 1e-4
    log_interval: int = 10
    log_all_for_first_epochs: int = 10
    shuffle_data: bool = True

    def __post_init__(self) -> None:
        """Ensure that the model path exists."""
        if not self.model_path.parent.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
