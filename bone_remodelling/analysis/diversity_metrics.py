
"""Compute simple dataset diversity metrics for biomechanical force profiles."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

from bone_remodelling.forward_data.reader import forward_data_reader

logger = logging.getLogger(__name__)

def compute_diversity_metrics(force_profiles: np.ndarray) -> dict[str, float]:
    """Compute simple dataset diversity metrics for biomechanical force profiles."""
    # Coverage fraction (fraction of nodes ever loaded)
    nonzero_counts = np.count_nonzero(force_profiles, axis=0)
    coverage_fraction = np.count_nonzero(nonzero_counts) / force_profiles.shape[1]

    # Shannon entropy of location distribution (normalized)
    # Average absolute load per node, normalised to sum 1
    node_loads = np.abs(force_profiles).sum(axis=0)
    normalized_node_loads = node_loads / node_loads.sum()
    location_entropy = entropy(normalized_node_loads)
    max_entropy = np.log(len(normalized_node_loads))
    normalized_entropy = location_entropy / max_entropy

    # Mean pairwise L2 distance (sampled for efficiency)
    n_samples = min(2000, force_profiles.shape[0])
    sample_indices = np.random.choice(force_profiles.shape[0], n_samples, replace=False)
    sampled = force_profiles[sample_indices]
    pairwise_differences = sampled[:, None, :] - sampled[None, :, :]
    l2 = np.linalg.norm(pairwise_differences, axis=-1)
    mean_pairwise_distance = np.mean(l2[np.triu_indices_from(l2, k=1)])

    # Force energy variance
    energies = np.sum(force_profiles**2, axis=1)
    energy_mean = np.mean(energies)
    energy_varience = np.var(energies)

    return {
        "coverage_fraction": coverage_fraction,
        "normalized_entropy": normalized_entropy,
        "mean_pairwise_distance": mean_pairwise_distance,
        "energy_mean": energy_mean,
        "energy_varience": energy_varience,
    }

def compute_diversity_metrics(force_profiles: np.ndarray) -> dict[str, Any]:
    """Compute simple dataset diversity metrics for biomechanical force profiles.
    
    Args:
    ----
        force_profiles (np.ndarray): Array of shape (N, M)
    """
    result = forward_data_reader()
    force_profiles, output_densities = np.array([]), np.array([])
    if result is not None:
        _, force_profiles, output_densities = result
        logger.info(f"Force Profiles Shape: {force_profiles.shape}")
        logger.info(f"Output Densities Shape: {output_densities.shape}")
        logger.info(f"Force Profiles Sample: {force_profiles[0]}")
        logger.info(f"Output Densities Sample: {output_densities[0]}")
    else:
        logger.error("Failed to load data: forward_data_reader returned None.")

    # Add a plot that visualizes the distribution of the output densities

    from bone_remodelling.forward_model.density_visualizer import (
        plot_density_matrix,
    )
    avg_output_densities = output_densities.mean(axis=0)
    avg_force_profiles = force_profiles.mean(axis=0)
    force_mask = np.ones_like(avg_force_profiles, dtype=bool)
    std_force_profiles = force_profiles.std(axis=0)
    std_output_densities = output_densities.std(axis=0)

    plot_density_matrix(
        matrix=avg_output_densities,
        force_data=(avg_force_profiles, force_mask),
        title="Location average of output densities - SL dataset",
        axis=plt.gca(),
    )
    plt.show()

    directory_path_triangular = Path("/home/gijs/Desktop/Thesis/data/raw/triangular/")
    result_triangular = forward_data_reader(directory_path_triangular)
    if result_triangular is not None:
        _, force_profiles_triangular, output_densities_triangular = result_triangular
    else:
        logger.error("Failed to load data from triangular dataset: forward_data_reader returned None.")

    force_profile_energy = np.zeros(force_profiles.shape[0])
    force_profile_flat = np.zeros((force_profiles.shape[0], force_profiles.shape[1]*force_profiles.shape[2]))
    for i in range(force_profiles.shape[0]):
        force_profile_energy[i] = np.square(force_profiles[i]).sum()
        force_profile_flat[i] = force_profiles[i].flatten()

    force_profile_energy_triangular = np.zeros(force_profiles_triangular.shape[0])
    force_profile_flat_triangular = np.zeros((force_profiles_triangular.shape[0], force_profiles_triangular.shape[1]*force_profiles_triangular.shape[2]))
    for i in range(force_profiles_triangular.shape[0]):
        force_profile_energy_triangular[i] = np.square(force_profiles_triangular[i]).sum()
        force_profile_flat_triangular[i] = force_profiles_triangular[i].flatten()

    plt.figure(1)
    plt.hist(force_profiles.flatten(), bins=200)
    plt.hist(force_profiles_triangular.flatten(), bins=200)
    plt.title("Distribution of Force Profiles")
    plt.yscale("log")
    plt.xlabel("Force Profile Value")
    plt.ylabel("Frequency")
    plt.legend(["Dataset Supervised Learning", "Dataset RL"])

    plt.figure(2)
    plt.hist(output_densities.flatten(), bins=200)
    plt.hist(output_densities_triangular.flatten(), bins=200)
    plt.title("Distribution of Output Densities")
    plt.yscale("log")
    plt.xlabel("Output Density Value")
    plt.ylabel("Frequency")
    plt.legend(["Dataset Supervised Learning", "Dataset RL"])

    plt.figure(3)
    plt.hist(force_profile_energy, bins=200)
    plt.hist(force_profile_energy_triangular, bins=200)
    plt.title("Distribution of Force Profile Energy")
    plt.yscale("log")
    plt.xlabel("Force Profile Energy Value")
    plt.ylabel("Frequency")
    plt.legend(["Dataset Supervised Learning", "Dataset RL"])
    plt.show()


    metrics_supervised = compute_diversity_metrics(force_profile_flat)
    metrics_rl = compute_diversity_metrics(force_profile_flat_triangular)

    logger.info("Metric | Supervised | RL")
    logger.info("--------------------------------------")
    for key in metrics_supervised:
        logger.info(f"{key:25s} | {metrics_supervised[key]:8.3f} | {metrics_rl[key]:8.3f}")
    return metrics_supervised

if __name__ == "__main__":
    compute_diversity_metrics()
