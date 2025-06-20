"""Advanced Neural Network Surrogate Model for Bone Remodeling Simulation."""

import torch
from torch import nn


class AdvancedNNSurrogateModel(nn.Module):
    """Advanced Neural Network Surrogate Model for Bone Remodeling Simulation.

    This model uses convolutional layers to process input data and predict bone density profiles.
    It includes methods for training, validation, saving, loading, and plotting loss history.

    Attributes
    ----------
        model (nn.Sequential): The neural network architecture.
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
        get_scheduler(optimizer, epochs):
            Get a learning rate scheduler.
        create_dataloader(X, y, batch_size=32, shuffle=True):
            Create a DataLoader for the dataset.

    """

    def __init__(self):
        """Initialize the AdvancedNNSurrogateModel."""
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1
            ),  # Input: (1, 3, 10)
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
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        """Forward pass through the model."""
        x = x.unsqueeze(1)  # reshape from (N, 3, 10) to (N, 1, 3, 10)
        out = self.model(x)
        return out.view(-1, 10, 10)  # reshape to (N, 10, 10)

    def save_model(self, file_path):
        """Save the model state to a file."""
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        """Load the model state from a file."""
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def __str__(self):
        """Model represented as String."""
        return f"AdvancedNNSurrogateModel(\n  {self.model}\n)"

    def __repr__(self):
        """Model represented as String."""
        return self.__str__()

    def __call__(self, x):
        """Call the forward method when the model is called."""
        return self.forward(x)

    def __len__(self):
        """Return the number of layers in the model."""
        return len(list(self.model))

    def plot_loss(self):
        """Plot training and validation loss history."""
        import matplotlib.pyplot as plt

        if not self.train_losses:
            print("No training history found.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")  # Set y-axis to logarithmic scale
        plt.title("Training and Validation Loss (Log Scale)")
        plt.legend()
        plt.grid(
            True, which="both", linestyle="--", linewidth=0.5
        )  # Improve grid visibility for log scale
        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_scheduler(optimizer, epochs):
        """Get a learning rate scheduler."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

    @staticmethod
    def create_dataloader(X, y, batch_size=32, shuffle=True):
        """Create a DataLoader for the dataset."""
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X.reshape(-1, 3, 10), dtype=torch.float32),
            torch.tensor(y.reshape(-1, 10, 10), dtype=torch.float32),
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
