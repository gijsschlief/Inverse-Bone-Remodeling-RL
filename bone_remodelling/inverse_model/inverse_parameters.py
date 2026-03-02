"""Stores parameters used for training the inverse model."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from bone_remodelling.surrogate_model.surrogate_parameters import TrainParameters

logger = logging.getLogger(__name__)


@dataclass
class InverseTrainParameters(TrainParameters):
    """Parameters for training the inverse model."""

    model_path: Path
    device: torch.device
    epochs: int = 1_000

    # Hyperparameters
    batch_size: int = 180
    weight_decay: float = 1e-5
    width: int = 2048
    encoder_depth: int = 4
    decoder_depth: int = 4
    dropout: float = 0.43

    # Learning rate variables
    learning_rate: float = 0.003
    patience_lr_scheduler: int = 10
    factor_lr_scheduler: float = 0.5
    cooldown_lr_scheduler: int = 0
    patience: int = 30
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
        minimal_encoder_depth: int = 2
        if self.encoder_depth < minimal_encoder_depth:
            logger.warning(
                f"Model encoder depth below buildable range, using minimun of {minimal_encoder_depth}.",
            )
        minimal_decoder_depth: int = 2
        if self.decoder_depth < minimal_decoder_depth:
            logger.warning(
                f"Model decoder depth below buildable range, using minimun of {minimal_decoder_depth}.",
            )
        minimal_width: int = 128
        if self.width < minimal_width:
            logger.warning(
                f"Model width below buildable range, using minimun of {minimal_width}.",
            )
