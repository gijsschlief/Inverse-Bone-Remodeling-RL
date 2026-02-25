"""Neural Network Surrogate Model for Bone Remodeling Simulation."""

import logging
from pathlib import Path

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

        self.input_fc = self._build_linear_encoder()
        self.conv_block = self._build_conv_decoder()

        self.train_losses: list[float] = []
        self.val_losses: list[torch.Tensor] = []

    def _build_linear_encoder(self) -> torch.nn.Sequential:
        """Build the encoder based on the width and depth of the model."""
        num_linear_layers = self.depth // 2
        linear_layers = []

        # Initial expansion
        linear_layers.append(torch.nn.Flatten())
        linear_layers.append(torch.nn.Linear(30, self.width // 2))
        linear_layers.append(torch.nn.ReLU())
        linear_layers.append(torch.nn.Linear(self.width // 2, self.width))
        linear_layers.append(torch.nn.ReLU())

        # Intermediate linear layers
        for _ in range(num_linear_layers - 3):
            linear_layers.append(torch.nn.Linear(self.width, self.width))
            linear_layers.append(torch.nn.ReLU())

        if self.dropout > 0:
            linear_layers.append(torch.nn.Dropout(self.dropout))

        # Project to spatial dimensions (128 channels at 5x5)
        self.spatial_ch = 128
        linear_layers.append(torch.nn.Linear(self.width, self.spatial_ch * 5 * 5))
        linear_layers.append(torch.nn.ReLU())

        return torch.nn.Sequential(*linear_layers)

    def _build_conv_decoder(self) -> torch.nn.Sequential:
        """Build the decoder based on the depth of the model."""
        num_conv_layers = self.depth // 2
        conv_layers = []

        # Layer 1: Upsample 5x5 -> 10x10
        conv_layers.append(
            torch.nn.ConvTranspose2d(
                self.spatial_ch,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        conv_layers.append(torch.nn.ReLU())
        conv_layers.append(torch.nn.BatchNorm2d(64))

        # Intermediate refinement layers (stay at 10x10)
        # We decrease channels progressively towards 1
        current_ch = 64
        for _ in range(num_conv_layers - 2):
            next_ch = max(32, current_ch // 2)
            conv_layers.append(
                torch.nn.Conv2d(current_ch, next_ch, kernel_size=3, padding=1),
            )
            conv_layers.append(torch.nn.ReLU())
            conv_layers.append(torch.nn.BatchNorm2d(next_ch))
            current_ch = next_ch

        # Final output layer to get 1 channel
        conv_layers.append(torch.nn.Conv2d(current_ch, 1, kernel_size=3, padding=1))
        return torch.nn.Sequential(*conv_layers)

    def update(self, width: int, depth: int, dropout: float) -> None:
        """Update the size of the neural network."""
        current_device = next(self.parameters()).device

        self.width = width
        self.depth = depth
        self.dropout = dropout

        self.input_fc = self._build_linear_encoder()
        self.conv_block = self._build_conv_decoder()
        self.to(current_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.input_fc(x)  # (N, 128*5*5)
        x = x.view(-1, 128, 5, 5)  # (N, 128, 5, 5)
        x = self.conv_block(x)  # (N, 1, 10, 10)
        return x.squeeze(1)  # (N, 10, 10)

    def save_model(self, file_path: Path) -> None:
        """Save the model state to a file."""
        torch.save(self.state_dict(), file_path)
