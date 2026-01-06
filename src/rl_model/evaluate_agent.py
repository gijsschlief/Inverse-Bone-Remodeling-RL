"""Evaluate the trained RL agent."""

import logging

import numpy as np
from bone_remodeling.src.forward_data.reader import forward_data_reader
from bone_remodeling.src.inverse_model.evaluate_inverse import (
    inverse_model_metrics,
    plot_inverse_model,
)
from bone_remodeling.src.rl_model.environment import BoneRemodelingEnvironment
from bone_remodeling.src.rl_model.forward_pass import (
    EnsembleForwarder,
    SurrogateForwarder,
)
from bone_remodeling.src.rl_model.parameters import RLParameters, RunConfiguration
from bone_remodeling.src.rl_model.render_callback import RenderCallback
from bone_remodeling.src.rl_model.reward_calculation import calculate_similarity
from bone_remodeling.src.surrogate_model.loader import SurrogateModelLoader
from bone_remodeling.src.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.src.surrogate_model.splitter import splitting
from matplotlib import pyplot as plt
from stable_baselines3 import PPO

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
    episode_rewards = []
    ssim_scores = []
    mse_errors = []
    all_force_profiles = []
    all_predicted_densities = []
    all_samples_forces = []
    all_sample_densities = []
    render_callback = RenderCallback()

    # For plotting worst sample at the end
    worst_sample_information = None
    worst_estimate_information = None
    worst_reward = float('inf')

    for ep in range(num_episodes):
        obs, _ = environment.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = environment.step(action)

            sample_information, estimate_information, plot_reward = environment.get_data_for_visualization()
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

    if worst_estimate_information is not None and worst_sample_information is not None and render:
        render_callback.render(
            sample_information=worst_sample_information,
            estimate_information=worst_estimate_information,
            reward=worst_reward)


    return {
        "rewards": episode_rewards,
        "ssim_scores": ssim_scores,
        "mse_errors": mse_errors,
        "sample_forces": all_samples_forces,
        "sample_densities": all_sample_densities,
        "forces": all_force_profiles,
        "predicted_densities": all_predicted_densities,
    }


def main() -> None:
    """Evaluate the RL agent."""
    run_parameters = RunConfiguration()

    result = forward_data_reader(run_parameters.data_path)
    if result is not None:
        _, target_forces, target_densities = result

    _, _, test_density_profiles, _, _, test_force_profiles = splitting(
        target_densities,
        target_forces,
        random_state=run_parameters.random_state,
    )

    model = PPO.load(run_parameters.agent_path.as_posix())

    rl_parameters = RLParameters()

    # USE THE SURROGATE OR ENSEMBLE FORWARDER DEPENDING ON THE TRAINING SETUP
    forwarder_surrogate = SurrogateForwarder(
    surrogate_model_path=run_parameters.surrogate_path,
    density_shape=test_density_profiles[0].shape,
    model_class=ReversedSurrogateModel,
    )

    forwarder_ensemble = EnsembleForwarder(  # noqa: F841
        model_paths=run_parameters.ensemble_path,
        model_class=ReversedSurrogateModel,
        model_loader=SurrogateModelLoader,
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
        num_episodes=len(test_density_profiles[:,1,1]),
        render=False,
    )

    # Force Magnitude Comparison
    test_peaks = np.max(np.abs(np.array(evaluation_result["sample_forces"])), axis=(1, 2))
    eval_peaks = np.max(np.abs(np.array(evaluation_result["forces"])), axis=(1, 2))
    eps = 1e-3
    test_peaks = np.clip(test_peaks, eps, None)
    eval_peaks = np.clip(eval_peaks, eps, None)
    plt.figure(num=2)
    plt.scatter(test_peaks, eval_peaks, alpha=0.4)
    max_test = test_peaks.max()
    max_eval = eval_peaks.max()
    min_test = test_peaks.min()
    min_eval = eval_peaks.min()
    max_val = max(max_test, max_eval)
    min_val = min(min_test, min_eval)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min_test, max_test)
    plt.ylim(min_eval, max_eval)
    plt.xlabel("True Peak Force")
    plt.ylabel("Predicted Peak Force")
    plt.title("Absolute Peak Force Magnitude Accuracy")
    plt.grid(visible=True)
    plt.show()

    sample_forces = np.array(evaluation_result["sample_forces"])
    predicted_forces = np.array(evaluation_result["forces"])
    sample_densities = np.array(evaluation_result["sample_densities"])
    predicted_densities = np.array(evaluation_result["predicted_densities"])
    inverse_model_metrics(sample_forces, predicted_forces)

    # Calculate average metrics
    avg_rewards = sum(evaluation_result["rewards"]) / len(evaluation_result["rewards"])
    avg_ssim = sum(evaluation_result["ssim_scores"]) / len(evaluation_result["ssim_scores"])
    avg_mse = sum(evaluation_result["mse_errors"]) / len(evaluation_result["mse_errors"])

    logger.info(f"Average Reward: {avg_rewards:.4f}, Average SSIM: {avg_ssim:.6f}, Average MSE: {avg_mse:.6f}")

    # Plot representative samples from evaluation
    for _ in range(100):
        k = np.random.randint(0, len(sample_forces))
        logger.info(f"Sample {k}: SSIM = {evaluation_result['ssim_scores'][k]:.6f}, MSE = {evaluation_result['mse_errors'][k]:.6f}")
        logger.info(f"Original Force Profile: {sample_forces[k]}")
        logger.info(f"Reconstructed Force Profile: {predicted_forces[k]}")
        plot_inverse_model(
            predicted_densities[k],
            sample_densities[k],
            predicted_forces[k],
            sample_forces[k],
        )
        plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
