"""Run a reinforcement learning environment for bone remodeling."""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from bone_remodelling.forward_data.force_profile_generator import ForceProfileGenerator
from bone_remodelling.forward_data.forward_data_manager import ForwardDataManager
from bone_remodelling.parameters import ConfigurationParameters
from bone_remodelling.rl_model.custum_actor import BoneFeaturesExtractor
from bone_remodelling.rl_model.environment import BoneRemodelingEnvironment
from bone_remodelling.rl_model.forward_pass import (
    FenicsForwarder,
    ForwardPass,
    SurrogateForwarder,
)
from bone_remodelling.rl_model.metrics import MetricsContainer
from bone_remodelling.rl_model.parameters import (
    RLParameters,
    RunConfiguration,
    TrainingStats,
)
from bone_remodelling.rl_model.render_callback import RenderCallback
from bone_remodelling.rl_model.reward_saving_callback import RewardSavingCallback
from bone_remodelling.rl_model.validation_callback import ValidationCallback
from bone_remodelling.rl_model.validation_environment_builder import (
    ValidationEnvironmentBuilder,
)
from bone_remodelling.surrogate_model.neural_network import SurrogateModel
from bone_remodelling.surrogate_model.splitter import load_and_split_data
from bone_remodelling.surrogate_model.surrogate_parameters import (
    SurrogateTrainParameters,
    TrainParameters,
)

logger = logging.getLogger(__name__)

# Container for the current learning rate, used to modify it during training.
current_learning_rate = {"value": RLParameters().learning_rate}


def save_model_safely(model: PPO, path: Path) -> Path:
    """Save the model to a file, ensuring no overwriting of existing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if Path.exists(path):
        directory = path.parent
        base_path = path.stem
        ext = path.suffix
        counter = 1
        while Path(f"{directory}/{base_path}_{counter}{ext}").exists():
            counter += 1
        path = Path(f"{directory}/{base_path}_{counter}{ext}")
    model.save(path)
    return Path(path)


def _build_environment(
    train_forces: np.ndarray,
    train_densities: np.ndarray,
    rl_parameters: RLParameters,
    forwarder: ForwardPass,
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
    while Path(f"{directory}/{base_path}_{counter}{ext}").exists():
        counter += 1
    if counter == 1:
        return path
    return Path(f"{directory}/{base_path}_{counter - 1}{ext}")


def load_training_stats(agent_path: Path) -> TrainingStats | None:
    """Load training statistics from a file corresponding to the agent."""
    stats_path = agent_path.with_suffix(".json")
    if stats_path.exists():
        try:
            with Path.open(stats_path, "r") as f:
                stats_data = json.load(f)
            return TrainingStats(**stats_data)
        except (json.JSONDecodeError, TypeError, OSError) as e:
            logger.warning(f"Failed to load training stats from {stats_path}: {e}")
    else:
        logger.info(f"No training stats found at {stats_path}. Starting fresh.")
    return None


def initialize_new_model(
    environment: VecNormalize,
    rl_parameters: RLParameters,
) -> PPO:
    """Initialize a new PPO model with the given environment.

    Args:
    ----
        environment (VecNormalize): The RL environment to use.
        rl_parameters (RLParameters): The parameters for the RL agent.

    Returns:
    -------
        PPO: A new PPO model instance.

    """
    logger.info("Initializing a new PPO model.")
    policy_kwargs = {
        "features_extractor_class": BoneFeaturesExtractor,
        "activation_fn": torch.nn.ReLU,
        "net_arch": {"pi": [1024, 1024, 512, 512], "vf": [1024, 1024, 512, 512]},
    }

    return PPO(
        policy="MlpPolicy",
        env=environment,
        verbose=rl_parameters.verbose,
        n_steps=rl_parameters.n_steps,
        batch_size=rl_parameters.batch_size,
        ent_coef=rl_parameters.ent_coef,
        learning_rate=learning_rate_container,
        n_epochs=rl_parameters.n_epochs,
        target_kl=rl_parameters.target_kl,
        max_grad_norm=rl_parameters.max_grad_norm,
        seed=rl_parameters.seed,
        device=rl_parameters.device,
        policy_kwargs=policy_kwargs,
    )


def train_rl_agent(
    config: ConfigurationParameters,
    run_parameters: RunConfiguration,
    rl_parameters: RLParameters,
    surrogate_parameters: TrainParameters,
) -> None:
    """Designs and trains a reinforcement learning agent for bone remodeling.

    This function initializes the bone remodeling environment, loads the surrogate model,
    and trains a reinforcement learning agent using the Proximal Policy Optimization (PPO)
    algorithm. It reads training data, splits it into training sets, and uses callbacks for
    rendering and saving rewards during training.

    Args:
    ----
        config (ConfigurationParameters): Configuration parameters for the run.
        run_parameters (RunConfiguration): Run-specific parameters.
        rl_parameters (RLParameters): Parameters for the RL agent.
        surrogate_parameters (TrainParameters): Parameters for training the surrogate model.

    """
    forward_data_manager = ForwardDataManager(run_parameters.data_path)
    (
        train_forces,
        validation_forces,
        _,
        train_densities,
        validation_densities,
        _,
    ) = load_and_split_data(
        forward_data_manager,
        random_state=run_parameters.random_state,
    )
    logger.info(f"Training RL agent on {len(train_densities)} samples.")

    metrics = MetricsContainer()

    forwarder: ForwardPass
    if run_parameters.forward_type == "fenics":
        forwarder = FenicsForwarder(
            config=config,
            force_profile=train_forces[0],
        )
    else:
        forwarder = SurrogateForwarder(
            config=config,
            model_class=SurrogateModel,
            train_parameters=surrogate_parameters,
        )

    validation_environment_builder = ValidationEnvironmentBuilder(
        forwarder,
        rl_parameters,
    )

    number_of_environments: int = run_parameters.number_of_environments
    base_seed = run_parameters.random_state
    logger.info(f"Using base_seed: {base_seed} for environment seeding.")

    make_environment = partial(
        _build_environment,
        train_forces=train_forces,
        train_densities=train_densities,
        rl_parameters=rl_parameters,
        forwarder=forwarder,
    )

    environment_functions = [
        lambda seed=base_seed + i: make_environment(seed=seed)
        for i in range(number_of_environments)
    ]
    vectorized_environment: VecNormalize = VecNormalize(
        SubprocVecEnv(environment_functions),
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
    )

    logger.info("Environment functions created, loading agent if it exists.")
    if run_parameters.agent_path.is_dir():
        model = initialize_new_model(vectorized_environment, rl_parameters)
        latest_agent_path = None
    else:
        latest_agent_path = find_latest_agent(run_parameters.agent_path)
        try:
            model = PPO.load(latest_agent_path, env=vectorized_environment)
            model.set_env(vectorized_environment)
            training_stats = load_training_stats(latest_agent_path)
            if training_stats is not None:
                model.learning_rate = training_stats.current_learning_rate
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
        callbacks: list[BaseCallback] = [
            RenderCallback(
                force_generator=ForceProfileGenerator(
                    (config.force_top_resolution, config.force_side_resolution),
                ),
                render_freq=run_parameters.render_frequency,
                environment_index=0,
                rl_parameters=rl_parameters,
            ),
        ]
        if run_parameters.reward_plot_path is not None:
            callbacks.append(
                RewardSavingCallback(
                    metrics=metrics,
                    out_path=run_parameters.reward_plot_path,
                ),
            )
        callbacks.append(
            ValidationCallback(
                metrics=metrics,
                learning_rate_container=current_learning_rate,
                validation_data=(
                    validation_forces,
                    validation_densities,
                ),
                validation_environment_builder=validation_environment_builder,
                run_config=run_parameters,
                rl_parameters=rl_parameters,
            ),
        )
        model.learn(
            total_timesteps=run_parameters.total_timesteps,
            callback=callbacks,
        )
        logger.info("Training complete.")
        saved_path = save_model_safely(model, run_parameters.agent_path)
        logger.info(f"Model saved to {saved_path}")
    finally:
        vectorized_environment.close()


def cli(config: ConfigurationParameters, remaining_args: list[str]) -> None:
    """Command-line interface for running the RL training."""
    parser = argparse.ArgumentParser(description="Generate density animations.")
    parser.add_argument(
        "--forward-type",
        choices=["fenics", "surrogate"],
        default="surrogate",
        help="Forward pass to use for the RL environment.",
    )
    args = parser.parse_args(remaining_args)
    run_parameters = RunConfiguration(
        output_dir=config.output_dir,
        forward_type=args.forward_type,
    )
    rl_parameters = RLParameters()
    model_path = (
        Path(config.output_dir) / Path("surrogate_models", "surrogate.pth")
    ).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate_parameters = SurrogateTrainParameters(model_path, device)
    train_rl_agent(config, run_parameters, rl_parameters, surrogate_parameters)
