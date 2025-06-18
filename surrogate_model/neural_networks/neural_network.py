import logging

import torch
from torch import nn


class NNSurrogateModel(nn.Module):
    def __init__(self):
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
