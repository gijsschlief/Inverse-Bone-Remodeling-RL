"""Neural Network Surrogate Model for Bone Remodeling Simulation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SurrogateModel(torch.nn.Module):
    """Surrogate Neural Network Model for bone remodeling simulation."""

    def __init__(self) -> None:
        """Initialize the InverseSurrogateModel."""
        super().__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
        )

        self.global_pool = torch.nn.AdaptiveAvgPool2d((3, 10))  # keep spatial shape fixed

        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),  # (N, 128, 3, 10) => (N, 128*3*10)
            torch.nn.Linear(128 * 3 * 10, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 30),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = x.unsqueeze(1)  # (N, 10, 10) → (N, 1, 10, 10)
        x = self.conv_block(x)  # → (N, 128, 10, 10)
        x = self.global_pool(x)  # → (N, 128, 10, 10)
        x = self.fc(x)  # → (N, 30)
        return x.view(-1, 3, 10)  # → (N, 3, 10)

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
