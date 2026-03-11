"""Implementation of a custom actor for the bone remodelling RL environment."""

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BoneFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for the bone remodelling RL environment."""

    def __init__(self, observation_space: spaces.Space) -> None:
        """Initialize the feature extractor."""
        # The input is 13x10 flattened = 130
        super().__init__(observation_space, features_dim=512)

        input_dimension: int = np.prod(observation_space.shape)
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dimension, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from the observations."""
        return self.net(observations)
