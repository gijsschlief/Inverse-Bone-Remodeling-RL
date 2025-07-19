"""Large Neural Network Surrogate Model for bone remodeling simulation."""

from typing import List

from torch import Tensor, nn

from bone_remodeling.src.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)


class LargeSurrogateModel(SurrogateModel):
    """Large Neural Network Surrogate Model for bone remodeling simulation."""

    def __init__(self) -> None:
        """Initialize the LargeSurrogateModel."""
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(
            (3, 10)
        )  # keep spatial shape fixed

        self.fc = nn.Sequential(
            nn.Flatten(),  # (N, 128, 3, 10) => (N, 128*3*10)
            nn.Linear(128 * 3 * 10, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 100),
        )

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = x.unsqueeze(1)  # (N, 3, 10) → (N, 1, 3, 10)
        x = self.conv_block(x)  # → (N, 128, 3, 10)
        x = self.global_pool(x)  # → (N, 128, 3, 10)
        x = self.fc(x)  # → (N, 100)
        return x.view(-1, 10, 10)  # → (N, 10, 10)
