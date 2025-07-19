"""Module for the reversed LargeSurrogateModel neural network."""

from torch import nn

from bone_remodeling.src.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)


class ReversedSurrogateModel(SurrogateModel):
    """Reversed Neural Network Surrogate Model for bone remodeling simulation."""

    def __init__(self) -> None:
        """Initialize the reversed ReversedSurrogateModel."""
        super().__init__()

        # Fully connected input block (no early dropout)
        self.input_fc = nn.Sequential(
            nn.Flatten(),  # (N, 3, 10) → (N, 30)
            nn.Linear(30, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 5 * 5),  # Prepare for upsampling
            nn.ReLU(),
            nn.Dropout(0.3),  # Only here, after features are richer
        )

        # Reshape to (N, 128, 5, 5) and upsample
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Final 10x10 map
        )
