"""Neural Network Surrogate Model for Bone Remodeling Simulation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bone_remodelling.surrogate_model.train_parameters import (
    SurrogateTrainParameters,
)

logger = logging.getLogger(__name__)


class SurrogateModel(torch.nn.Module):
    """Surrogate Neural Network Model for bone remodeling simulation."""

    def __init__(self) -> None:
        """Initialize the SurrogateModel."""
        super().__init__()

        # Fully connected input block (no early dropout)
        self.input_fc = torch.nn.Sequential(
            torch.nn.Flatten(),  # (N, 3, 10) → (N, 30)
            torch.nn.Linear(30, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 128 * 5 * 5),  # Prepare for upsampling
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),  # Only here, after features are richer
        )

        # Reshape to (N, 128, 5, 5) and upsample
        self.conv_block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Final 10x10 map
        )

        self.train_losses: list[float] = []
        self.val_losses: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.input_fc(x)  # (N, 128*5*5)
        x = x.view(-1, 128, 5, 5)  # (N, 128, 5, 5)
        x = self.conv_block(x)  # (N, 1, 10, 10)
        return x.squeeze(1)  # (N, 10, 10)

    def save_model(self, file_path: Path) -> None:
        """Save the model state to a file."""
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        """Load the model state from a file."""
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return f"LargeSurrogateModel(\n  {self.input_fc}\n  {self.conv_block}\n)"

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return self.__str__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call the model with input tensor."""
        return self.forward(x)

    def __len__(self) -> int:
        """Return the number of layers in the model."""
        return len(list(self.input_fc)) + len(list(self.conv_block))

    def plot_loss(self) -> None:
        """Plot the training and validation loss history."""
        if not self.train_losses:
            logger.error("No training history found.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        if self.val_losses:
            float_val_losses = [loss.item() for loss in self.val_losses]
            plt.plot(float_val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Training and Validation Loss (Log Scale)")
        plt.legend()
        plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_scheduler(
        optimizer: torch.optim.Optimizer,
        train_parameters: SurrogateTrainParameters,
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Get a learning rate scheduler for the surrogate model.

        Args:
        ----
            optimizer (torch.optim.Optimizer): The optimizer to schedule.
            train_parameters (SurrogateTrainParameters): Training parameters including device, epochs, batch size, learning rate, patience, min delta, log interval, and log all for first epochs.

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

    @staticmethod
    def create_dataloader(
        input_features: np.ndarray | torch.Tensor,
        output_labels: np.ndarray | torch.Tensor,
        batch_size: int = 32,
        *,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader for the surrogate model.

        Args:
        ----
            input_features (np.ndarray | torch.Tensor): Input features of shape (N, 3, 10).
            output_labels (np.ndarray | torch.Tensor): Output labels of shape (N, 10, 10).
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
        -------
            torch.utils.data.DataLoader: DataLoader for the surrogate model.

        """
        if isinstance(input_features, torch.Tensor):
            input_tensor = input_features.clone().detach()
        else:
            input_tensor = torch.tensor(
                input_features.reshape(-1, 3, 10),
                dtype=torch.float32,
            )

        if isinstance(output_labels, torch.Tensor):
            y_tensor = output_labels.clone().detach()
        else:
            y_tensor = torch.tensor(
                output_labels.reshape(-1, 10, 10),
                dtype=torch.float32,
            )

        dataset = torch.utils.data.TensorDataset(input_tensor, y_tensor)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
