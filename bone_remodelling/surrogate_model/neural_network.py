"""Neural Network Surrogate Model for Bone Remodeling Simulation."""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SurrogateModel(torch.nn.Module):
    """Surrogate Neural Network Model for bone remodeling simulation."""

    def __init__(self, width: int = 1024, depth: int = 6, dropout: float = 0.3) -> None:
        """Initialize the SurrogateModel."""
        super().__init__()
        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.max_channel = self.width // 16

        self.input_fc = self._build_linear_encoder()
        self.conv_block = self._build_conv_decoder()

        self.train_losses: list[float] = []
        self.val_losses: list[torch.Tensor] = []

    def _build_linear_encoder(self) -> torch.nn.Sequential:
        """Build the encoder based on the width and depth of the model."""
        num_linear_layers = self.depth // 2
        linear_layers = []
        linear_layers.append(torch.nn.Flatten())

        widths = np.linspace(30, self.width, num_linear_layers).astype(int)

        # Linear layers
        for i in range(len(widths) - 1):
            linear_layers.append(torch.nn.Linear(widths[i], widths[i+1]))
            linear_layers.append(torch.nn.ReLU())

        if self.dropout > 0:
            linear_layers.append(torch.nn.Dropout(self.dropout))

        # Project to spatial dimensions (width/16) channels at 4x4)
        linear_layers.append(torch.nn.Linear(self.width, self.max_channel * 4 * 4))
        linear_layers.append(torch.nn.ReLU())

        return torch.nn.Sequential(*linear_layers)

    def _build_conv_decoder(self) -> torch.nn.Sequential:
        """Build the decoder based on the depth of the model."""
        num_conv_layers = self.depth // 2
        conv_layers = []
        minimal_channel = 1
        next_channel = max(minimal_channel, self.max_channel // 2)

        # Layer 1: Upsample 4x4 -> 10x10
        conv_layers.append(
            torch.nn.ConvTranspose2d(
                self.max_channel,
                next_channel,
                kernel_size=4,
                stride=2,
            ),
        )
        conv_layers.append(torch.nn.ReLU())
        conv_layers.append(torch.nn.BatchNorm2d(next_channel))

        # Inbetween layers dependent on the depth
        current_channel = next_channel
        for _ in range(num_conv_layers - 2):
            next_channel = max(minimal_channel, current_channel // 2)
            conv_layers.append(
                torch.nn.Conv2d(current_channel, next_channel, kernel_size=3, padding=1),
            )
            conv_layers.append(torch.nn.ReLU())
            conv_layers.append(torch.nn.BatchNorm2d(next_channel))
            current_channel = next_channel

        # Final output layer to get 1 channel
        conv_layers.append(torch.nn.Conv2d(current_channel, 1, kernel_size=3, padding=1))
        return torch.nn.Sequential(*conv_layers)

    def update(self, width: int, depth: int, dropout: float) -> None:
        """Update the size of the neural network."""
        current_device = next(self.parameters()).device

        self.width = width
        self.depth = depth
        self.dropout = dropout
        self.max_channel = self.width // 16

        self.input_fc = self._build_linear_encoder()
        self.conv_block = self._build_conv_decoder()
        self.to(current_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.input_fc(x)  # (N, max_channel*4*4)
        x = x.view(-1, self.max_channel, 4, 4)
        x = self.conv_block(x)  # (N, 1, 10, 10)
        return x.squeeze(1)  # (N, 10, 10)

    def save_model(self, file_path: Path) -> None:
        """Save the model state to a file."""
        torch.save(self.state_dict(), file_path)
