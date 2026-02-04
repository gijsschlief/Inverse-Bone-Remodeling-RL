"""Simple Neural Network Inverse Model for Bone Remodeling Simulation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


class InverseModel(torch.nn.Module):
    """Inverse Neural Network Model for bone remodeling simulation."""

    def __init__(self, input_size: int=100, hidden_dimension: int=256, output_dim: int=3) -> None:
        """Initialize the inverse model.

        Args:
        ----
            input_size (int): Flattened 10x10 density map (100).
            hidden_dimension (int): Hidden layer width.
            output_dim (int): Number of predicted force components (e.g., 3).

        """
        super().__init__()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension*2, hidden_dimension*4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension*4, hidden_dimension*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension*2, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the inverse model."""
        sides = 3
        if x.ndim == sides:
            x = x.view(x.size(0), -1)
        return self.fc(x)

    def save_model(self, file_path: Path) -> None:
        """Save the model state to a file."""
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        """Load the model state from a file."""
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return f"InverseModel(\n  {self.fc})"

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return self.__str__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call the model with input tensor."""
        return self.forward(x)

    def __len__(self) -> int:
        """Return the number of layers in the model."""
        return len(list(self.fc))

    def plot_loss(self) -> None:
        """Plot the training and validation loss history."""
        if not self.train_losses:
            logger.error("No training history found.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss")
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
        epochs: int,  # noqa: ARG004
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Get a learning rate scheduler for the surrogate model.

        Args:
        ----
            optimizer (torch.optim.Optimizer): The optimizer to schedule.
            epochs (int): The total number of training epochs.

        Returns:
        -------
            torch.optim.lr_scheduler.ReduceLROnPlateau: The learning rate scheduler.

        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
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
