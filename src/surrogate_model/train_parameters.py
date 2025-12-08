"""Stores parameters used for training the surrogate model."""

from dataclasses import dataclass

import torch


@dataclass
class SurrogateTrainParameters:
    """Parameters for training the surrogate model."""

    device: torch.device
    epochs: int = 1_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience_lr_scheduler: int = 10
    factor_lr_scheduler: float = 0.5
    patience: int = 20
    min_delta: float = 1e-4
    log_interval: int = 10
    log_all_for_first_epochs: int = 10
