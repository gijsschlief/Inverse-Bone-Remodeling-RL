"""Evaluate the trained RL agent."""

import logging
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import PPO

from bone_remodelling.forward_data.force_profile_generator import ForceProfileGenerator
from bone_remodelling.forward_data.forward_data_manager import ForwardDataManager
from bone_remodelling.forward_model.density_visualizer import plot_density_matrix
from bone_remodelling.parameters import ConfigurationParameters
from bone_remodelling.rl_model.environment import BoneRemodelingEnvironment
from bone_remodelling.rl_model.forward_pass import (
    SurrogateForwarder,
)
from bone_remodelling.rl_model.parameters import RLParameters, RunConfiguration
from bone_remodelling.rl_model.render_callback import RenderCallback
from bone_remodelling.rl_model.reward_calculation import calculate_similarity
from bone_remodelling.surrogate_model.neural_network import SurrogateModel
from bone_remodelling.surrogate_model.splitter import splitting
from bone_remodelling.surrogate_model.surrogate_parameters import (
    SurrogateTrainParameters,
)

logger = logging.getLogger(__name__)


def evaluate_agent(
    model: PPO,
    environment: BoneRemodelingEnvironment,
    num_episodes: int = 10,
    *,
    render: bool = False,
) -> dict[str, list[np.ndarray] | list[float]]:
    """Evaluate the trained RL agent.

    Args:
    ----
        model: The trained Stable-Baselines3 model.
        environment: The Gym environment to evaluate in.
        num_episodes (int): Number of evaluation episodes.
        render (bool): Whether to render the environment during evaluation.

    Returns:
    -------
        dict: Evaluation metrics per episode.

    """
    force_generator = ForceProfileGenerator()
    episode_rewards = []
    ssim_scores = []
    mse_errors = []
    all_force_profiles = []
    all_predicted_densities = []
    all_samples_forces = []
    all_sample_densities = []
    render_callback = RenderCallback(force_generator)

    # For plotting worst sample at the end
    worst_sample_information = None
    worst_estimate_information = None
    worst_reward = float("inf")

    sample_information, estimate_information, plot_reward = (
        environment.get_data_for_visualization()
    )
    for ep in range(num_episodes):
        obs, _ = environment.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = environment.step(action)

            sample_information, estimate_information, plot_reward = (
                environment.get_data_for_visualization()
            )
            if render:
                render_callback.render(
                    sample_information=sample_information,
                    estimate_information=estimate_information,
                    reward=plot_reward,
                )

            total_reward += float(reward)
            done = terminated or truncated

        # Track worst performing sample
        if worst_reward > plot_reward:
            logger.info(f"New worst sample found with reward: {plot_reward:.4f}")
            worst_reward = plot_reward
            worst_sample_information = sample_information
            worst_estimate_information = estimate_information

        # Evaluation metrics
        episode_rewards.append(total_reward)
        ssim = calculate_similarity(
            reference_matrix=sample_information[2],
            comparison_matrix=estimate_information[2],
            method="ssim",
        )
        ssim_scores.append(ssim)

        mse = calculate_similarity(
            reference_matrix=sample_information[2],
            comparison_matrix=estimate_information[2],
            method="mse",
        )
        mse_errors.append(mse)
        all_force_profiles.append(estimate_information[1])
        all_predicted_densities.append(estimate_information[2])
        all_samples_forces.append(sample_information[1])
        all_sample_densities.append(sample_information[2])

        logger.info(
            f"Episode {ep + 1}/{num_episodes} - Total Reward: {total_reward:.4f}, Final SSIM: {ssim:.6f}, Final MSE: {mse:.6f}",
        )

    if (
        worst_estimate_information is not None
        and worst_sample_information is not None
        and render
    ):
        render_callback.render(
            sample_information=worst_sample_information,
            estimate_information=worst_estimate_information,
            reward=worst_reward,
        )

    return {
        "rewards": episode_rewards,
        "ssim_scores": ssim_scores,
        "mse_errors": mse_errors,
        "sample_forces": all_samples_forces,
        "sample_densities": all_sample_densities,
        "forces": all_force_profiles,
        "predicted_densities": all_predicted_densities,
    }


def run_agent_evaluation(config: ConfigurationParameters) -> None:
    """Evaluate the RL agent."""
    run_parameters = RunConfiguration(output_dir=config.output_dir)

    forward_data_manager = ForwardDataManager(
        (config.output_dir / "raw").resolve(),
    )
    result = forward_data_manager.load_directory()
    if result is None:
        raise ValueError("Failed to load forward data from directory")
    _, target_forces, target_densities = result

    _, _, test_density_profiles, _, _, test_force_profiles = splitting(
        target_densities,
        target_forces,
        random_state=run_parameters.random_state,
    )

    model = PPO.load(run_parameters.agent_path.as_posix())

    rl_parameters = RLParameters()

    surrogate_parameters = SurrogateTrainParameters(
        model_path=(
            config.output_dir / Path("surrogate_models", "surrogate.pth")
        ).resolve(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    forwarder_surrogate = SurrogateForwarder(
        config=config,
        model_class=SurrogateModel,
        train_parameters=surrogate_parameters,
    )

    agent_evaluation_environment = BoneRemodelingEnvironment(
        forwarder=forwarder_surrogate,
        target_densities=test_density_profiles,
        target_forces=test_force_profiles,
        rl_parameters=rl_parameters,
    )
    evaluation_result = evaluate_agent(
        model,
        agent_evaluation_environment,
        num_episodes=len(test_density_profiles[:, 1, 1]),
        render=False,
    )

    sample_forces = np.array(evaluation_result["sample_forces"])
    predicted_forces = np.array(evaluation_result["forces"])
    sample_densities = np.array(evaluation_result["sample_densities"])
    predicted_densities = np.array(evaluation_result["predicted_densities"])

    # Plot distribution of SSIM scores
    figures_dir = config.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(
        evaluation_result["ssim_scores"],
        bins=30,
        color="skyblue",
        edgecolor="black",
    )
    plt.title("Distribution of SSIM Scores across Test Set")
    plt.xlabel("SSIM")
    plt.ylabel("Frequency")
    plt.savefig(figures_dir / Path("ssim_distribution.png"))
    plt.close()

    # Calculate average metrics
    avg_rewards = sum(evaluation_result["rewards"]) / len(evaluation_result["rewards"])
    avg_ssim = sum(evaluation_result["ssim_scores"]) / len(
        evaluation_result["ssim_scores"],
    )
    avg_mse = sum(evaluation_result["mse_errors"]) / len(
        evaluation_result["mse_errors"],
    )

    logger.info(
        f"Average Reward: {avg_rewards:.4f}, Average SSIM: {avg_ssim:.6f}, Average MSE: {avg_mse:.6f}",
    )

    # Plot representative samples from evaluation
    force_generator = ForceProfileGenerator()
    force_mask = force_generator.generate_force_mask()
    for _ in range(20):
        k = np.random.randint(0, len(sample_forces))
        logger.info(
            f"Sample {k}: SSIM = {evaluation_result['ssim_scores'][k]:.6f}, MSE = {evaluation_result['mse_errors'][k]:.6f}",
        )
        logger.info(f"Original Force Profile: {sample_forces[k]}")
        logger.info(f"Reconstructed Force Profile: {predicted_forces[k]}")
        plot_inverse_model(
            predicted_densities[k],
            sample_densities[k],
            (predicted_forces[k], force_mask),
            (sample_forces[k], force_mask),
        )
        plt.savefig(figures_dir / Path(f"eval_sample_{k}.png"))
        plt.close()


def plot_inverse_model(
    predicted_density: np.ndarray,
    sample_density: np.ndarray,
    predicted_force: tuple[np.ndarray, np.ndarray],
    sample_force: tuple[np.ndarray, np.ndarray],
) -> None:
    """Plot the predicted vs sample density and force profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_density_matrix(
        matrix=predicted_density,
        title="Predicted Density",
        axis=axes[0],
        color_scale=(0, 1),
        force_data=predicted_force,
    )
    plot_density_matrix(
        matrix=sample_density,
        title="Sample Density",
        axis=axes[1],
        color_scale=(0, 1),
        force_data=sample_force,
    )


def cli(
    configuration_parameters: ConfigurationParameters,
) -> None:
    """CLI entry point for evaluating the RL agent."""
    run_agent_evaluation(configuration_parameters)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = ConfigurationParameters(
        output_dir=Path("output"),
    )
    run_agent_evaluation(config)
