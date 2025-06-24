"""Large Neural Network Surrogate Model for bone remodeling simulation."""

import logging
from typing import List

import numpy as np
import torch


class LargeSurrogateModel(torch.nn.Module):
    """Large Neural Network Surrogate Model for bone remodeling simulation."""

    def __init__(self) -> None:
        """Initialize the LargeSurrogateModel."""
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

        self.global_pool = torch.nn.AdaptiveAvgPool2d(
            (3, 10)
        )  # keep spatial shape fixed

        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),  # (N, 128, 3, 10) => (N, 128*3*10)
            torch.nn.Linear(128 * 3 * 10, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 100),
        )

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = x.unsqueeze(1)  # (N, 3, 10) → (N, 1, 3, 10)
        x = self.conv_block(x)  # → (N, 128, 3, 10)
        x = self.global_pool(x)  # → (N, 128, 3, 10)
        x = self.fc(x)  # → (N, 100)
        return x.view(-1, 10, 10)  # → (N, 10, 10)

    def save_model(self, file_path: str) -> None:
        """Save the model state to a file."""
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        """Load the model state from a file."""
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return f"LargeSurrogateModel(\n  {self.conv_block}\n  {self.fc}\n)"

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return self.__str__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call the model with input tensor x."""
        return self.forward(x)

    def __len__(self) -> int:
        """Return the total number of layers in the model."""
        return len(list(self.conv_block)) + len(list(self.fc))

    def plot_loss(self) -> None:
        """Plot the training and validation loss over epochs."""
        import matplotlib.pyplot as plt

        if not self.train_losses:
            logging.info("No training history found.")
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
    def get_scheduler(optimizer: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Create a learning rate scheduler for the surrogate model."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

    @staticmethod
    def create_dataloader(
        x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader for the surrogate model."""
        if isinstance(x, torch.Tensor):
            x_tensor = x.clone().detach()
        else:
            x_tensor = torch.tensor(x.reshape(-1, 3, 10), dtype=torch.float32)

        if isinstance(y, torch.Tensor):
            y_tensor = y.clone().detach()
        else:
            y_tensor = torch.tensor(y.reshape(-1, 10, 10), dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
