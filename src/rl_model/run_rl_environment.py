"""Run a reinforcement learning environment for bone remodeling."""

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from bone_remodeling.src.rl_model.environment import BoneRemodellingEnvironment
from bone_remodeling.src.rl_model.forward_pass import (
    EnsembleForwarder,  # noqa: F401
    FenicsForwarder,  # noqa: F401
    SurrogateForwarder,  # noqa: F401
)
from bone_remodeling.src.rl_model.parameters import RLParameters
from bone_remodeling.src.rl_model.render_callback import RenderCallback
from bone_remodeling.src.rl_model.reward_saving_callback import RewardSavingCallback
from bone_remodeling.src.surrogate_model.loader import SurrogateModelLoader
from bone_remodeling.src.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,  # noqa: F401
)
from bone_remodeling.src.surrogate_model.splitter import load_and_split_data

logger = logging.getLogger(__name__)


def save_model_safely(model: PPO, path: Path) -> Path:
    """Save the model to a file, ensuring no overwriting of existing files."""
    if Path.exists(path):
        base_path = path.stem
        ext = path.suffix
        counter = 1
        while Path(f"{base_path}_{counter}{ext}").exists():
            counter += 1
        path = Path(f"{base_path}_{counter}{ext}")
    model.save(path)
    return Path(path)


def find_latest_agent(path: Path) -> Path:
    """Find the latest agent file in the specified directory."""
    directory = path.parent
    base_path = path.stem
    ext = path.suffix
    logger.info(f"Directory: {directory}, Base: {base_path}, Ext: {ext}")
    counter = 1
    # Check for existing files and increment the counter until a unique name is found
    while Path(f"{directory}/{base_path}_{counter}{ext}").exists():
        counter += 1

    return Path(f"{directory}/{base_path}_{counter - 1}{ext}")


def main(agent_path: Path, data_path: Path, surrogate_path: Path | list[Path]) -> None:
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
    (
        train_forces,
        _,
        _,
        train_densities,
        _,
        _,
    ) = load_and_split_data(data_path, random_state=0)
    logger.info(f"Training RL agent on {len(train_densities)} samples.")

    rl_parameters = RLParameters()

    #forwarder = SurrogateForwarder(surrogate_model_path=surrogate_path, density_shape=train_densities[0].shape, model_class=ReversedSurrogateModel)
    #forwarder = FenicsForwarder(force_profile=train_forces[0], initial_density_field=np.ones(train_densities[0].shape) * 0.8)
    forwarder = EnsembleForwarder(model_paths=surrogate_path, model_class=ReversedSurrogateModel, model_loader=SurrogateModelLoader)

    remodeling_environment = BoneRemodellingEnvironment(
        forwarder=forwarder,
        target_densities=train_densities,
        target_forces=train_forces,
        rl_parameters=rl_parameters,
    )

    logger.info("Environment created,loading agent if it exists.")
    if agent_path.is_dir():
        model = PPO("MlpPolicy", remodeling_environment, verbose=1)
        latest_agent_path = None
    else:
        latest_agent_path = find_latest_agent(agent_path)
        model = PPO.load(latest_agent_path, env=remodeling_environment)
        model.set_env(remodeling_environment)

    logger.info(
        f"Starting training with agent at {latest_agent_path if latest_agent_path is not None else 'new model'}.",
    )

    model.learn(
        total_timesteps=1_000,
        callback=[
            RenderCallback(render_freq=1),
            RewardSavingCallback(
                out_path="/home/gijs/Desktop/Thesis/data/figures/reward_curve_RL_special.png",
            ),
        ],
    )
    logger.info("Training complete.")

    saved_path = save_model_safely(model, agent_path)
    logger.info(f"Model saved to {saved_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    AGENT_PATH = Path("/home/gijs/Desktop/Thesis/data/agents/agents.zip")
    DATA_PATH = Path(
        "/home/gijs/Desktop/Thesis/data/raw/training_triangular_third_order_1000_samples_0720_1430.json",
    )
    SURROGATE_PATH = Path("/home/gijs/Desktop/Thesis/data/models/trained_model_3.pth")
    SURROGATE_PATHS = [
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_4.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_5.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_6.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_7.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_8.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_9.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_10.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_11.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_12.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_13.pth"),
    ]
    main(agent_path=AGENT_PATH, data_path=DATA_PATH, surrogate_path=SURROGATE_PATHS)
