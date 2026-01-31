"""Pretraining script for the RL model using supervised learning."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3 as sb3
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bone_remodelling.forward_data.reader import forward_data_reader
from bone_remodelling.forward_data.visualizer import plot_density_matrix
from bone_remodelling.rl_model.environment import BoneRemodelingEnvironment
from bone_remodelling.rl_model.validate_pretrained_agent import (
    validate_pretrained_agent,
)
from bone_remodelling.surrogate_model.splitter import splitting

logger = logging.getLogger(__name__)


def main() -> None:
    """Supervised pretraining of the rl agent."""
    directory_path = Path("/home/gijs/Desktop/Thesis/data/raw/")
    result = forward_data_reader(directory_path)
    if result is not None:
        _, target_forces, target_densities = result

    remodeling_environment = BoneRemodelingEnvironment(
        surrogate_model_path=Path(
            "/home/gijs/Desktop/Thesis/data/models/trained_model_1.pth",
        ),
        target_densities=target_densities[0],
        target_forces=target_forces[0],
        max_steps=100,
    )

    model = sb3.PPO("MlpPolicy", remodeling_environment, verbose=1)

    # Access policy network
    policy_net = model.policy
    actor = policy_net.action_net

    # Freeze everything except actor (optional)
    for param in policy_net.value_net.parameters():
        param.requires_grad = False

    # Setup supervised training
    split_data_result = splitting(
        input_features=target_densities,
        output_features=target_forces,
        random_state=0,
    )
    if split_data_result is None:
        logger.error("Data splitting failed. Exiting.")
        return

    (
        train_densities,
        validation_densities,
        _,
        train_forces,
        validation_forces,
        _,
    ) = split_data_result

    # Flatten densities per sample, for example if densities are 10x10:
    density_flat = train_densities.reshape(train_densities.shape[0], -1)
    force_flat = train_forces.reshape(train_forces.shape[0], -1)

    density_tensor = torch.tensor(density_flat, dtype=torch.float32)
    force_tensor = torch.tensor(force_flat, dtype=torch.float32)
    dataset = TensorDataset(density_tensor, force_tensor)

    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(20):
        total_loss = 0.0
        for xb, yb in loader:
            pred = model.policy(xb)[0]
            pred = pred.view(pred.size(0), -1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch + 1} - Loss: {total_loss / len(loader):.6f}")

    # Save model
    torch.save(
        model.policy.state_dict(),
        "/home/gijs/Desktop/Thesis/data/pretrained_agents/trained_agent_weights.pth",
    )

    validation_estimated_force_profiles = validate_pretrained_agent(
        agent=model,
        density_profiles=validation_densities,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    for _ in range(3):
        random_index = np.random.randint(0, len(validation_densities))
        _, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_density_matrix(
            matrix=validation_densities[random_index],
            force_profile=validation_estimated_force_profiles[random_index],
            title="Estimated Force Profile",
            axis=axes[0],
        )

        plot_density_matrix(
            matrix=validation_densities[random_index],
            force_profile=validation_forces[random_index],
            title="Target Force Profile",
            axis=axes[1],
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
