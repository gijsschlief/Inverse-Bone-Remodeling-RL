"""Visualize force profiles on top of density matrices."""

import logging

import matplotlib.pyplot as plt
import numpy as np

from bone_remodelling.forward_data.force_profile_generator import (
    ForceProfileGenerator,
)

logger = logging.getLogger(__name__)


def visualise_profiles(
    force_profiles: np.ndarray, force_profile_energy: np.ndarray
) -> None:
    """Visualise generated force profiles."""
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.hist(force_profiles.flatten(), bins=200)
    plt.title("Distribution of Force Profiles")
    plt.yscale("log")
    plt.xlabel("Force Profile Value")
    plt.ylabel("Frequency")

    # Logarithmic histogram of energy values
    plt.subplot(2, 2, 2)
    bins = np.logspace(-2, 5, 200).tolist()
    plt.hist(force_profile_energy, bins=bins)
    plt.title("Distribution of Force Profile Energy")
    plt.xscale("log")
    plt.xlabel("Force Profile Energy Value")
    plt.ylabel("Frequency")

    # Test location distribution of forces
    plt.subplot(2, 2, 3)
    mean = np.zeros(force_profiles.shape[1] * force_profiles.shape[2])
    std = np.zeros(force_profiles.shape[1] * force_profiles.shape[2])
    for i in range(force_profiles.shape[1]):
        for k in range(force_profiles.shape[2]):
            mean[i * force_profiles.shape[2] + k] = np.mean(force_profiles[:, i, k])
            std[i * force_profiles.shape[2] + k] = np.std(force_profiles[:, i, k])
    plt.errorbar(
        np.arange(force_profiles.shape[1] * force_profiles.shape[2]),
        mean,
        yerr=std,
        fmt="o",
        ecolor="red",
        capsize=2,
    )
    plt.title("Mean and Std Dev of Force Profile Locations")
    plt.xlabel("Force Profile Index")
    plt.ylabel("Mean Force Value")
    plt.show()


def analyse_force_generator() -> None:
    """Generate and visualise force profiles."""
    generator = ForceProfileGenerator(
        profile_top_and_sides=(10, 10),
        force_bounds=(0.1, 10.0),
        energy_bounds=(1e2, 5e4),
        batch_seed=42,
    )
    force_profiles = generator.merger(num_samples=100_000)
    # force_profiles = generator.triangular_only(num_samples=100_000)
    force_profile_energy = np.sum(force_profiles**2, axis=(1, 2))

    # Find empty profiles
    empty_profiles = np.where(force_profile_energy == 0)[0]
    if len(empty_profiles) > 0:
        logger.warning(
            f"Found {len(empty_profiles)} empty force profiles at indices: {empty_profiles}"
        )

    logger.info(f"Generated profiles like: {force_profiles[0]} and {force_profiles[1]}")
    max_force_value = np.max(np.abs(force_profiles))
    logger.info(f"Maximum force value across all profiles: {max_force_value}")

    visualise_profiles(force_profiles, force_profile_energy)


if __name__ == "__main__":
    analyse_force_generator()
