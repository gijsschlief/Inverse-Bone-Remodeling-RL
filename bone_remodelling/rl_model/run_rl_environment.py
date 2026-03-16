"""Run a reinforcement learning environment for bone remodeling."""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import argparse
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
    validation_data = (validation_forces, validation_densities)
    logger.info(f"Training RL agent on {len(train_densities)} samples.")

    metrics = MetricsContainer()

    forwarder: ForwardPass = _build_forwarder(
        config,
        run_parameters,
        train_forces,
        surrogate_parameters,
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
    stats_dict = None  # Ensure stats_dict is always defined
    if run_parameters.agent_path.is_dir():
        model = initialize_new_model(vectorized_environment, rl_parameters)
        latest_agent_path = None

    else:
        latest_agent_path = find_latest_agent(run_parameters.agent_path)
        try:
            model = PPO.load(latest_agent_path, env=vectorized_environment)
            model.set_env(vectorized_environment)

            stats_dict = getattr(model, "custom_stats", None)
            if stats_dict:
                metrics.resume_from_history(stats_dict)
                current_learning_rate["value"] = stats_dict.get(
                    "current_learning_rate",
                    rl_parameters.learning_rate,
                )
                vectorized_environment.env_method(
                    "set_curriculum_complexity",
                    stats_dict.get("current_complexity", 1),
                )

                logger.info(
                    f"Resumed from {latest_agent_path} at Level {stats_dict['current_complexity']}",
                )

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
            total_timesteps=run_parameters.total_timesteps,
            callback=_build_callbacks(
                config,
                run_parameters,
                rl_parameters,
                metrics,
                current_learning_rate,
                forwarder,
                validation_data,
                stats_dict,
            ),
            reset_num_timesteps=False,
        )
    finally:
        vectorized_environment.close()


def _build_forwarder(
    config: ConfigurationParameters,
    run_parameters: RunConfiguration,
    train_forces: np.ndarray,
    surrogate_parameters: TrainParameters,
) -> ForwardPass:
    if run_parameters.forward_type == "fenics":
        return FenicsForwarder(
            config=config,
            force_profile=train_forces[0],
        )
    return SurrogateForwarder(
        config=config,
        model_class=SurrogateModel,
        train_parameters=surrogate_parameters,
    )


def _build_callbacks(
    config: ConfigurationParameters,
    run_parameters: RunConfiguration,
    rl_parameters: RLParameters,
    metrics: MetricsContainer,
    current_learning_rate: dict[str, float],
    forwarder: ForwardPass,
    validation_data: tuple[np.ndarray, np.ndarray],
    stats_dict: dict | None = None,
) -> list[BaseCallback]:
    """Build the list of callbacks for training."""
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

    validation_environment_builder = ValidationEnvironmentBuilder(
        forwarder,
        rl_parameters,
    )
    validation_callback = ValidationCallback(
        metrics=metrics,
        learning_rate_container=current_learning_rate,
        validation_data=validation_data,
        validation_environment_builder=validation_environment_builder,
        run_config=run_parameters,
        rl_parameters=rl_parameters,
    )
    if stats_dict:
        validation_callback.best_ssim = stats_dict.get("max_ssim", -np.inf)
        validation_callback.patience_counter = stats_dict.get("current_patience", 0)
        validation_callback.current_complexity = stats_dict.get("current_complexity", 1)

    callbacks.append(validation_callback)
    return callbacks


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
