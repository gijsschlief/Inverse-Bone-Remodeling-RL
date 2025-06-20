"""Neural Network Surrogate Model for Bone Remodeling Simulation."""

import logging

import torch
from torch import nn


class NNSurrogateModel(nn.Module):
    """Neural Network Surrogate Model for Bone Remodeling Simulation.

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
        save_model(file_path):
            Save the model state to a file.
        load_model(file_path):
            Load the model state from a file.
        plot_loss():
            Plot training and validation loss history.

    """

    def __init__(self):
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
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        """Forward pass through the model."""
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits.view(-1, 10, 10)  # Reshape output to 10x10

    def save_model(self, file_path):
        """Save the model to the specified file path."""
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        """Load the model from the specified file path."""
        self.load_state_dict(torch.load(file_path))
        self.eval()  # Set the model to evaluation mode after loading

    def __str__(self):
        """Return a string representation of the model architecture."""
        return f"NeuralNetwork(\n  {self.linear_relu_stack}\n)"

    def __repr__(self):
        """Return a detailed string representation of the model."""
        return f"NeuralNetwork(\n  {self.linear_relu_stack}\n)"

    def __call__(self, x):
        """Call the forward method of the model."""
        return self.forward(x)

    def __len__(self):
        """Return the number of layers in the model."""
        return len(self.linear_relu_stack)

    def plot_loss(self):
        """Plot training and validation loss history."""
        import matplotlib.pyplot as plt

        if not self.train_losses:
            logging.error("No training losses recorded.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
