"""Stores parameters used for training the surrogate model."""

from dataclasses import dataclass

import torch


@dataclass
class SurrogateTrainParameters:
    """Parameters for training the surrogate model."""

    device: torch.device
    epochs: int = 10_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience_lr_scheduler: int = 25
    factor_lr_scheduler: float = 0.5
    cooldown_lr_scheduler: int = 10
    patience: int = 500
    min_delta: float = 1e-4
    log_interval: int = 10
    log_all_for_first_epochs: int = 10
