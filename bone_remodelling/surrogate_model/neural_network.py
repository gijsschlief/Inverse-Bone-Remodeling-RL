"""Neural Network Surrogate Model for Bone Remodeling Simulation."""

import logging
from pathlib import Path

import torch

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
