"""Evaluate the trained RL agent."""

import logging
from pathlib import Path

from gymnasium import Env
from stable_baselines3 import PPO

from bone_remodeling.src.forward_data.reader import forward_data_reader
from bone_remodeling.src.rl_model.environment import BoneRemodelingEnvironment
from bone_remodeling.src.rl_model.forward_pass import SurrogateForwarder
from bone_remodeling.src.rl_model.parameters import RLParameters, RunConfiguration
from bone_remodeling.src.rl_model.reward_calculation import calculate_similarity
from bone_remodeling.src.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.src.surrogate_model.splitter import splitting

logger = logging.getLogger(__name__)


def evaluate_agent(
    model: PPO,
    environment: Env,
    num_episodes: int = 10,
    *,
    render: bool = False,
) -> dict:
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
    mse_errors = []
    all_force_profiles = []
    all_predicted_densities = []

    for ep in range(num_episodes):
        obs, _ = environment.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = environment.step(action)

            total_reward += float(reward)
            done = terminated or truncated

            if render:
                environment.render()

        # Evaluation metrics for this episode
        episode_rewards.append(total_reward)
        mse = calculate_similarity(
            reference_matrix=environment.target_densities,
            comparison_matrix=info["predicted_density"],
            method="ssim",
        )
        mse_errors.append(mse)
        all_force_profiles.append(info["force_profile"])
        all_predicted_densities.append(info["predicted_density"])

        logger.info(
            f"Episode {ep + 1}/{num_episodes} - Total Reward: {total_reward:.4f}, Final MSE: {mse:.6f}",
        )

    return {
        "rewards": episode_rewards,
        "mse_errors": mse_errors,
        "forces": all_force_profiles,
        "predicted_densities": all_predicted_densities,
    }


def main() -> None:
    """Evaluate the RL agent."""
    run_parameters = RunConfiguration()

    result = forward_data_reader(run_parameters.data_path)
    if result is not None:
        _, target_forces, target_densities = result

    _, _, test_density_profiles, _, _, test_forces = splitting(
        target_densities,
        target_forces,
        random_state=run_parameters.random_state,
    )

    model = PPO.load("/home/gijs/Desktop/Thesis/data/agents/surrogate_agent_10mil.zip")

    rl_parameters = RLParameters()

    forwarder_surrogate = SurrogateForwarder(
    surrogate_model_path=run_parameters.surrogate_path,
    density_shape=test_density_profiles[0].shape,
    model_class=ReversedSurrogateModel,
    )

    all_results = []
    for i in range(len(test_density_profiles)):
        agent_evaluation_environment = BoneRemodelingEnvironment(
            forwarder=forwarder_surrogate,
            target_densities=test_density_profiles[i],
            target_forces=test_forces[i],
            rl_parameters=rl_parameters,
        )
        evaluation_result = evaluate_agent(
            model,
            agent_evaluation_environment,
            num_episodes=1,
            render=True,
        )
        all_results.append(evaluation_result)

    # Calculate average metrics
    avg_rewards = sum(res["rewards"][0] for res in all_results) / len(all_results)
    avg_mse = sum(res["mse_errors"][0] for res in all_results) / len(all_results)

    logger.info(f"Average Reward: {avg_rewards:.4f}, Average SSIM: {avg_mse:.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
