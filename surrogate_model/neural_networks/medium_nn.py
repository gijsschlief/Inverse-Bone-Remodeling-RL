"""Medium Neural Network Surrogate Model for Bone Remodeling Simulation."""

from typing import List

import numpy as np
import torch


class MediumSurrogateModel(torch.nn.Module):
    """Medium Neural Network Surrogate Model for bone remodeling simulation.

    This model uses a convolutional neural network architecture to predict
    bone density matrices based on input features. It is designed to handle
    input data with a shape of (N, 3, 10) and outputs density matrices of shape (N, 10, 10).
    The model includes methods for training, saving, loading, and visualizing the training history.
    It also provides a method to create a DataLoader for training and validation datasets.

    Attributes
    ----------
        model (nn.Sequential): The neural network architecture.
        train_losses (List[float]): List to store training losses over epochs.
        val_losses (List[float]): List to store validation losses over epochs.

    Methods
    -------
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the model.
        save_model(file_path: str) -> None:
            Save the model state to a file.
        load_model(file_path: str) -> None:
            Load the model state from a file.

    """

    def __init__(self) -> None:
        """Initialize the AdvancedNNSurrogateModel with a neural network architecture."""
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 3 * 10, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 100),
        )
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = x.unsqueeze(1)  # reshape from (N, 3, 10) to (N, 1, 3, 10)
        out = self.model(x)
        return out.view(-1, 10, 10)  # reshape to (N, 10, 10)

    def save_model(self, file_path: str) -> None:
        """Save the model state to a file."""
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        """Load the model state from a file."""
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return f"AdvancedNNSurrogateModel(\n  {self.model}\n)"

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return self.__str__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call the model with input tensor x."""
        return self.forward(x)

    def __len__(self) -> int:
        """Return the number of layers in the model."""
        return len(list(self.model))

    def plot_loss(self) -> None:
        """Plot the training and validation loss over epochs."""
        import matplotlib.pyplot as plt

        if not self.train_losses:
            print("No training history found.")
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
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_scheduler(
        optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Create a learning rate scheduler for the surrogate model."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

    @staticmethod
    def create_dataloader(
        X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader for the surrogate model."""
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X.reshape(-1, 3, 10), dtype=torch.float32),
            torch.tensor(y.reshape(-1, 10, 10), dtype=torch.float32),
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
