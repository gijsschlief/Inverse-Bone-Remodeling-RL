"""Neural Network Surrogate Model for Bone Remodeling Simulation."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class InverseModel(torch.nn.Module):
    """Inverse Neural Network Model for bone remodeling simulation."""

    def __init__(self) -> None:
        """Initialize the inverse model."""
        super().__init__()

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        # === Coordinate channels: encode (x, y) position ===
        self.register_buffer(
            "coord_x",
            torch.linspace(-1, 1, 10).repeat(10, 1).unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "coord_y",
            torch.linspace(-1, 1, 10).repeat(10, 1).t().unsqueeze(0).unsqueeze(0),
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
            torch.nn.Sigmoid(),
        )

        # === Feature projection ===
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 10 * 10, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, 31),
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
