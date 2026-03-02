"""Scheduler adapting the learning rate during training of the surrogate model."""

import torch

from bone_remodelling.surrogate_model.surrogate_parameters import (
    TrainParameters,
)


def build_optimizer(
    model: torch.nn.Module,
    train_parameters: TrainParameters,
) -> torch.optim.AdamW:
    """Get an optimizer for the surrogate model."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_parameters.learning_rate,
        weight_decay=train_parameters.weight_decay,
    )


def build_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    train_parameters: TrainParameters,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Get a learning rate scheduler for the surrogate model.

    Args:
    ----
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        train_parameters (TrainParameters): Training parameters including device, epochs, batch size, learning rate, patience, min delta, log interval, and log all for first epochs.

    Returns:
    -------
        torch.optim.lr_scheduler.ReduceLROnPlateau: The learning rate scheduler.

    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_parameters.factor_lr_scheduler,
        patience=train_parameters.patience_lr_scheduler,
        cooldown=train_parameters.cooldown_lr_scheduler,
    )


def build_dataloader(
    input_features: torch.Tensor,
    output_labels: torch.Tensor,
    batch_size: int = 32,
    *,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the surrogate model.

    Args:
    ----
        input_features (torch.Tensor): Input features of shape (N, 3, 10).
        output_labels (torch.Tensor): Output labels of shape (N, 10, 10).
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
    -------
        torch.utils.data.DataLoader: DataLoader for the surrogate model.

    """
    input_tensor = input_features.clone().detach()
    output_tensor = output_labels.clone().detach()
    dataset = torch.utils.data.TensorDataset(input_tensor, output_tensor)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
