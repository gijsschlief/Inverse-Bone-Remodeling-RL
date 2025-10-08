"""Neural Network Inverse Model for Bone Remodeling Simulation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


class InverseModel(torch.nn.Module):
    """Inverse Neural Network Model for bone remodeling simulation."""

    def __init__(self) -> None:
        """Initialize the inverse model."""
        super().__init__()

        self.train_losses = []
        self.val_losses = []

        # === Coordinate channels: encode (x, y) position ===
        self.register_buffer(
            "coord_x",
            torch.linspace(-1, 1, 10).repeat(10, 1).unsqueeze(0).unsqueeze(0)
        )
        self.register_buffer(
            "coord_y",
            torch.linspace(-1, 1, 10).repeat(10, 1).t().unsqueeze(0).unsqueeze(0)
        )

        # === Encoder ===
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
        )

        # === Residual block ===
        self.res_block = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
        )

        # === Spatial Attention ===
        self.attention = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, kernel_size=1),
            torch.nn.Identity(),
        )

        # === Feature projection ===
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 10 * 10, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the inverse model."""
        # x: (N, 10, 10)
        n = x.shape[0]
        coord_x = self.coord_x.repeat(n, 1, 1, 1)
        coord_y = self.coord_y.repeat(n, 1, 1, 1)
        x = x.unsqueeze(1)  # (N, 1, 10, 10)
        x = torch.cat([x, coord_x, coord_y], dim=1)  # (N, 3, 10, 10)

        features = self.encoder(x)
        residual = features
        features = self.res_block(features) + residual  # Residual connection

        # Apply spatial attention
        attn = self.attention(features)
        features = features * attn  # weighted features

        # Flatten and map to outputs
        out = features.flatten(1)
        return self.fc(out)

    def save_model(self, file_path: Path) -> None:
        """Save the model state to a file."""
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        """Load the model state from a file."""
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return f"LargeSurrogateModel(\n  {self.fc}\n  {self.encoder}\n  {self.res_block}\n  {self.attention}\n)"

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return self.__str__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call the model with input tensor."""
        return self.forward(x)

    def __len__(self) -> int:
        """Return the number of layers in the model."""
        return len(list(self.fc)) + len(list(self.encoder)) + len(list(self.res_block)) + len(list(self.attention))

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
