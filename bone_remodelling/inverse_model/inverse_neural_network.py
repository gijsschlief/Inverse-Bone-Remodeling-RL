"""Neural Network Surrogate Model for Bone Remodeling Simulation."""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ResidualBlock(torch.nn.Module):
    """Residual Block for the convolutional encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the Residual Block."""
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

        self.shortcut = torch.nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1),
                torch.nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Residual Block."""
        return self.relu(self.conv(x) + self.shortcut(x))


class InverseModel(torch.nn.Module):
    """Inverse Neural Network Model for bone remodeling simulation."""

    def __init__(self, width: int = 1024, depth: int = 6, dropout: float = 0.3) -> None:
        """Initialize the SurrogateModel."""
        super().__init__()
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.max_channel = self.width // 16

        self.register_buffer(
            "coord_x",
            torch.linspace(-1, 1, 10).repeat(10, 1).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "coord_y",
            torch.linspace(-1, 1, 10).repeat(10, 1).t().unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

        self.conv_encoder = self._build_conv_encoder()
        self.output_fc = self._build_linear_decoder()

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _build_conv_encoder(self) -> torch.nn.Sequential:
        """Build the encoder based on the width and depth of the model."""
        num_conv_layers = self.depth // 2
        conv_layers = []

        current_channel = 3  # Start with 3 channels (density + x + y)
        next_channel = (
            max(4, self.max_channel // (2 ** (num_conv_layers - 2)))
            if num_conv_layers > 1
            else self.max_channel
        )

        for _ in range(num_conv_layers - 1):
            conv_layers.append(ResidualBlock(current_channel, next_channel))
            current_channel = next_channel
            next_channel = min(self.max_channel, current_channel * 2)

        conv_layers.append(
            torch.nn.Conv2d(
                current_channel,
                self.max_channel,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
        )
        conv_layers.append(torch.nn.ReLU())
        conv_layers.append(torch.nn.BatchNorm2d(self.max_channel))
        return torch.nn.Sequential(*conv_layers)

    def _build_linear_decoder(self) -> torch.nn.Sequential:
        """Build the decoder based on the depth of the model."""
        num_linear_layers = self.depth // 2
        linear_layers = []

        linear_layers.append(torch.nn.Flatten())
        linear_layers.append(torch.nn.Linear(self.max_channel * 4 * 4, self.width))
        linear_layers.append(torch.nn.ReLU())

        if self.dropout > 0:
            linear_layers.append(torch.nn.Dropout(self.dropout))

        widths = np.linspace(self.width, 31, num_linear_layers).astype(int)

        for i in range(len(widths) - 1):
            linear_layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                linear_layers.append(torch.nn.ReLU())
        return torch.nn.Sequential(*linear_layers)

    def update(self, width: int, depth: int, dropout: float) -> None:
        """Update the size of the neural network dynamically."""
        current_device = next(self.parameters()).device

        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.max_channel = self.width // 16

        self.conv_encoder = self._build_conv_encoder()
        self.output_fc = self._build_linear_decoder()
        self.to(current_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Density Image -> Conv Encoder -> Linear Decoder -> Parameters."""
        n = x.shape[0]
        coord_x = self.coord_x.repeat(n, 1, 1, 1)
        coord_y = self.coord_y.repeat(n, 1, 1, 1)

        x = x.unsqueeze(1)
        x = torch.cat([x, coord_x, coord_y], dim=1)

        x = self.conv_encoder(x)
        return self.output_fc(x)

    def save_model(self, file_path: Path) -> None:
        """Save the model state to a file."""
        torch.save(self.state_dict(), file_path)
