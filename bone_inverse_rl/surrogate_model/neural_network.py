from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * 10, 128),  # Input layer (3x10 flattened to 30)
            nn.ReLU(),
            nn.Linear(128, 256),    # Hidden layer 1
            nn.ReLU(),
            nn.Linear(256, 512),    # Hidden layer 2
            nn.ReLU(),
            nn.Linear(512, 10 * 10) # Output layer (10x10 flattened to 100)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits.view(-1, 10, 10)  # Reshape output to 10x10