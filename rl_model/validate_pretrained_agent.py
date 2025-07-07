"""Validate pretrained agent to test improved performance."""

import logging

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def validate_pretrained_agent(
    agent,
    density_profiles: np.ndarray,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> np.ndarray:
    """Calculate the estimated forces of the pretrained RL agent on validation data.

    This function calculates performance of a pretrained RL agent in predicting force profiles.

    Args:
    ----
        agent: pretrained RL agent (e.g., stable_baselines3 PPO)
        density_profiles: validation density matrices (shape: [num_samples, height, width])
        device: torch device

    Returns:
    -------
        np.ndarray: predicted forcec profiles

    """
    num_samples = density_profiles.shape[0]
    if num_samples == 0:
        logging.error("Validation data is empty. Cannot validate agent.")
        raise ValueError("Validation data is empty. Cannot validate agent.")
    predicted_forces = []

    for i in range(num_samples):
        observation_matrix = density_profiles[i].astype(np.float32)
        action, _ = agent.predict(observation_matrix, deterministic=True)
        action_tensor = torch.tensor(action, dtype=torch.float32).to(device)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)  # batch dimension

        predicted_forces.append(action_tensor.cpu().numpy())

    return np.array(predicted_forces)
