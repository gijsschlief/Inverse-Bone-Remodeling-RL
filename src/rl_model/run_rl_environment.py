"""Run a reinforcement learning environment for bone remodeling."""

import logging
from functools import partial
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from bone_remodeling.src.rl_model.environment import BoneRemodelingEnvironment
from bone_remodeling.src.rl_model.forward_pass import (
    EnsembleForwarder,
    FenicsForwarder,
    ForwardPass,
    SurrogateForwarder,
)
from bone_remodeling.src.rl_model.parameters import RLParameters
from bone_remodeling.src.rl_model.render_callback import RenderCallback
from bone_remodeling.src.rl_model.reward_saving_callback import RewardSavingCallback
from bone_remodeling.src.rl_model.validation_callback import ValidationCallback
from bone_remodeling.src.rl_model.validation_environment_builder import (
    ValidationEnvironmentBuilder,
)
from bone_remodeling.src.surrogate_model.loader import SurrogateModelLoader
from bone_remodeling.src.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.src.surrogate_model.splitter import load_and_split_data

logger = logging.getLogger(__name__)

# Container for the current learning rate, used to modify it during training.
current_learning_rate = {"value": RLParameters().learning_rate}


def save_model_safely(model: PPO, path: Path) -> Path:
    """Save the model to a file, ensuring no overwriting of existing files."""
    if Path.exists(path):
        directory = path.parent
        base_path = path.stem
        ext = path.suffix
        counter = 1
        while Path(f"{directory}/{base_path}_{counter}{ext}").exists():
            counter += 1
        path = Path(f"{directory}/{base_path}_{counter}{ext}")
    model.save(path)
    #TODO: ALSO SAVE THE NORMALIZATION PARAMETERS!
    return Path(path)


def _build_environment(
    train_forces: np.ndarray,
    train_densities: np.ndarray,
    rl_parameters: RLParameters,
    forwarder: type[ForwardPass],
    seed: int,
) -> BoneRemodelingEnvironment:
    """Build a bone remodeling environment for reinforcement learning."""
    environment = BoneRemodelingEnvironment(
        forwarder=forwarder,
        target_densities=train_densities,
        target_forces=train_forces,
        rl_parameters=rl_parameters,
    )
    environment.reset(seed=seed)
    return environment


def learning_rate_container(progress_remaining: float) -> float:  # noqa: ARG001
    """Container for the learning rate, used to modify it during training."""
    return current_learning_rate["value"]


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
    if counter == 1:
        # If no files exist, return the original path
        return path

    return Path(f"{directory}/{base_path}_{counter - 1}{ext}")


def initialize_new_model(
    environment: BoneRemodelingEnvironment,
    rl_parameters: RLParameters,
) -> PPO:
    """Initialize a new PPO model with the given environment.

    Args:
    ----
        environment (BoneRemodelingEnvironment): The RL environment to use.
        rl_parameters (RLParameters): The parameters for the RL agent.

    Returns:
    -------
        PPO: A new PPO model instance.

    """
    logger.info("Initializing a new PPO model.")
    return PPO(
        policy="MlpPolicy",
        env=environment,
        verbose=rl_parameters.verbose,
        n_steps=rl_parameters.n_steps,
        batch_size=rl_parameters.batch_size,
        ent_coef=rl_parameters.ent_coef,
        learning_rate=learning_rate_container,
        seed=rl_parameters.seed,
        device=rl_parameters.device,
    )


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
        validation_forces,
        _,
        train_densities,
        validation_densities,
        _,
    ) = load_and_split_data(data_path, random_state=0)
    logger.info(f"Training RL agent on {len(train_densities)} samples.")

    rl_parameters = RLParameters()

    forwarder_surrogate = SurrogateForwarder(  # noqa: F841
        surrogate_model_path=Path(
            "/home/gijs/Desktop/Thesis/data/models/trained_model.pth",
        ),
        density_shape=train_densities[0].shape,
        model_class=ReversedSurrogateModel,
    )
    forwarder_fenics = FenicsForwarder(
        force_profile=train_forces[0],
        initial_density_field=np.ones(train_densities[0].shape) * 0.8,
    )

    #forwarder_ensemble = EnsembleForwarder(
    #    model_paths=surrogate_path,
    #    model_class=ReversedSurrogateModel,
    #    model_loader=SurrogateModelLoader,
    #)
    validation_environment_builder = ValidationEnvironmentBuilder(
        forwarder_surrogate, rl_parameters,
    )

    number_of_environments: int = 10
    base_seed = 0
    logger.info(f"Using base_seed: {base_seed} for environment seeding.")

    make_environment = partial(
        _build_environment,
        train_forces=train_forces,
        train_densities=train_densities,
        rl_parameters=rl_parameters,
        forwarder=forwarder_surrogate,
    )

    environment_functions = [
        partial(make_environment, seed=base_seed + i)
        for i in range(number_of_environments)
    ]
    vectorized_environment = SubprocVecEnv(environment_functions)
    vectorized_environment = VecNormalize(
        vectorized_environment,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
    )

    logger.info("Environment functions created, loading agent if it exists.")
    if agent_path.is_dir():
        model = initialize_new_model(vectorized_environment, rl_parameters)
        latest_agent_path = None
    else:
        latest_agent_path = find_latest_agent(agent_path)
        try:
            model = PPO.load(latest_agent_path, env=vectorized_environment)
            model.set_env(vectorized_environment)
        except FileNotFoundError:
            logger.warning(
                f"Agent file {latest_agent_path} not found. Starting with a new model.",
            )
            model = initialize_new_model(vectorized_environment, rl_parameters)
        model.set_env(vectorized_environment)

    logger.info(
        f"Starting training with agent at {latest_agent_path if latest_agent_path is not None else 'new model'}.",
    )

    try:
        model.learn(
            total_timesteps=1_000_000,
            callback=[
                RenderCallback(
                    render_freq=10_000,
                    environment_index=0,
                    rl_parameters=rl_parameters,
                ),
                RewardSavingCallback(
                    out_path="/home/gijs/Desktop/Thesis/data/figures/reward_curve_RL_discrete.png",
                ),
                ValidationCallback(
                    learning_rate_container=current_learning_rate,
                    validation_data=(validation_forces[:40], validation_densities[:40]),
                    validation_environment_builder=validation_environment_builder,
                    final_forwarder=forwarder_fenics,
                    rl_parameters=rl_parameters,
                    validation_frequency=200_000,
                ),
            ],
        )
        logger.info("Training complete.")
        saved_path = save_model_safely(model, agent_path)
        logger.info(f"Model saved to {saved_path}")
    finally:
        vectorized_environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    AGENT_PATH = Path("/home/gijs/Desktop/Thesis/data/agents/surrogate_agent_1mil.zip")
    DATA_PATH = Path(
        "/home/gijs/Desktop/Thesis/data/raw/triangular/",
    )
    SURROGATE_PATHS = [
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_1.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_2.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_3.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_4.pth"),

    ]
    main(agent_path=AGENT_PATH, data_path=DATA_PATH, surrogate_path=SURROGATE_PATHS)
