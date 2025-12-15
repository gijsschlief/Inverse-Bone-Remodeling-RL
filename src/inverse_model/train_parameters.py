"""Stores parameters used for training the inverse surrogate model."""

from dataclasses import dataclass

import torch


@dataclass
class InverseTrainParameters:
    """Parameters for training the inverse model."""

    device: torch.device
    epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience: int = 25
    min_delta: float = 1e-4
    log_interval: int = 10
    log_all_for_first_epochs: int = 10
