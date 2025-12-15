"""Small Neural Network Surrogate Model for Bone Remodeling Simulation."""

from torch import Tensor, nn

from bone_remodeling.src.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)


class SmallSurrogateModel(SurrogateModel):
    """Small Neural Network Surrogate Model for Bone Remodeling Simulation.

    This model uses a fully connected architecture to process input data and predict bone density profiles.
    It includes methods for training, validation, saving, loading, and plotting loss history.

    Attributes
    ----------
        train_losses (list): List to store training losses.
        val_losses (list): List to store validation losses.

    Methods
    -------
        forward(x):
            Forward pass through the model.

    """

    def __init__(self) -> None:
        """Initialize the NNSurrogateModel."""
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * 10, 128),  # Input layer (3x10 flattened to 30)
            nn.ReLU(),
            nn.Linear(128, 256),  # Hidden layer 1
            nn.ReLU(),
            nn.Linear(256, 512),  # Hidden layer 2
            nn.ReLU(),
            nn.Linear(512, 10 * 10),  # Output layer (10x10 flattened to 100)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model."""
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits.view(-1, 10, 10)  # Reshape output to 10x10
