"""Run a reinforcement learning environment for bone remodeling."""

import logging
import os
from pathlib import Path

from stable_baselines3 import PPO

from bone_remodeling.src.forward_data.reader import forward_data_reader
from bone_remodeling.src.rl_model.environment import BoneRemodellingEnvironment
from bone_remodeling.src.rl_model.render_callback import RenderCallback
from bone_remodeling.src.rl_model.reward_saving_callback import RewardSavingCallback
from bone_remodeling.src.surrogate_model.splitter import splitting


def save_model_safely(model: PPO, path: Path) -> Path:
    """Save the model to a file, ensuring no overwriting of existing files."""
    if os.path.exists(path):
        base_path, ext = os.path.splitext(path)
        counter = 1
        while os.path.exists(f"{base_path}_{counter}{ext}"):
            counter += 1
        path = Path(f"{base_path}_{counter}{ext}")
    model.save(path)
    return Path(path)


def find_latest_agent(path: Path) -> Path:
    """Find the latest agent file in the specified directory."""
    if os.path.exists(path):
        base_path, ext = os.path.splitext(path)
        counter = 1
    while os.path.exists(f"{base_path}_{counter}{ext}"):
        counter += 1
    return Path(f"{base_path}_{counter-1}{ext}")


def main(agent_path: Path, data_path: Path, surrogate_path: Path) -> None:
    """Designs and trains a reinforcement learning agent for bone remodeling.

    This function initializes the bone remodeling environment, loads the surrogate model,
    and trains a reinforcement learning agent using the Proximal Policy Optimization (PPO)
    algorithm. It reads training data, splits it into training sets, and uses callbacks for
    rendering and saving rewards during training.

    Args:
    ----
        agent_path (Path): Path to the agent model file.
        data_path (Path): Path to the data file for training.
        surrogate_path (Path): Path to the surrogate model file.

    """
    result = forward_data_reader(data_path)
    assert result is not None, "Failed to read data from the specified path."
    _, target_forces, target_densities = result

    train_densities, _, _, train_forces, _, _ = splitting(
        target_densities, target_forces, random_state=0
    )
    logging.info(f"Training RL agent on {len(train_densities)} samples.")

    remodeling_environment = BoneRemodellingEnvironment(
        surrogate_model_path=surrogate_path,
        target_densities=train_densities,
        target_forces=train_forces,
        max_steps=100,
    )

    logging.info("Environment created,loading agent if it exists.")
    if agent_path is None:
        model = PPO("MlpPolicy", remodeling_environment, verbose=1)
    else:
        latest_agent_path = find_latest_agent(agent_path)
        if latest_agent_path is None:
            model = PPO("MlpPolicy", remodeling_environment, verbose=1)
        else:
            model = PPO.load(latest_agent_path, env=remodeling_environment)
            model.set_env(remodeling_environment)

    logging.info(
        f"Starting training with agent at {latest_agent_path if latest_agent_path else 'new model'}."
    )

    model.learn(
        total_timesteps=1000,
        callback=[
            RenderCallback(render_freq=1),
            RewardSavingCallback(
                out_path="/home/gijs/Desktop/Thesis/data/figures/reward_curve_RL_special.png"
            ),
        ],
    )
    logging.info("Training complete.")

    saved_path = save_model_safely(model, agent_path)
    logging.info(f"Model saved to {saved_path}")


if __name__ == "__main__":
    AGENT_PATH = Path("/home/gijs/Desktop/Thesis/data/agents/trained_agent_special.zip")
    DATA_PATH = Path(
        "/home/gijs/Desktop/Thesis/data/raw/training_triangular_profiles_10000_samples_0708_1959.json"
    )
    SURROGATE_PATH = Path("/home/gijs/Desktop/Thesis/data/models/trained_model_1.pth")
    main(agent_path=AGENT_PATH, data_path=DATA_PATH, surrogate_path=SURROGATE_PATH)
