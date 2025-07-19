"""Medium Neural Network Surrogate Model for Bone Remodeling Simulation."""

from torch import Tensor, nn

from bone_remodeling.src.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)


class MediumSurrogateModel(SurrogateModel):
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

    """

    def __init__(self) -> None:
        """Initialize the MediumSurrogateModel with a neural network architecture."""
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(32 * 3 * 10, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 100),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = x.unsqueeze(1)  # reshape from (N, 3, 10) to (N, 1, 3, 10)
        out = self.model(x)
        return out.view(-1, 10, 10)  # reshape to (N, 10, 10)
