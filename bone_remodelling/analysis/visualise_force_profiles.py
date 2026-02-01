"""Visualize force profiles on top of density matrices."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bone_remodelling.forward_data.force_profile_generator import (
    ForceProfileGenerator,
)
from bone_remodelling.forward_data.reader import forward_data_reader  # type: ignore
from bone_remodelling.forward_model.density_visualizer import (
    plot_density_matrix,  # type: ignore
)

logger = logging.getLogger(__name__)

def visualise_profiles(force_profiles: np.ndarray, force_profile_energy: np.ndarray) -> None:
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
            mean[i*force_profiles.shape[2] + k] = np.mean(force_profiles[:, i, k])
            std[i*force_profiles.shape[2] + k] = np.std(force_profiles[:, i, k])
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

def visualize_force_comparison(
    force_profiles_1: np.ndarray,
    force_profiles_2: np.ndarray,
    label_1: str = "New",
    label_2: str = "Original",
    title: str = "Force Profile Comparison",
) -> None:
    """Visualizes the given data using matplotlib.

    Args:
    ----
        force_profiles_1 (np.ndarray): Force profiles from the first dataset.
        force_profiles_2 (np.ndarray): Force profiles from the second dataset.
        label_1 (str): Label for the first dataset.
        label_2 (str): Label for the second dataset.
        title (str): Title of the plot.

    """

    def _preprocess_force(force_profile: np.ndarray) -> np.ndarray:
        sides_with_forces = 3
        force_profile = np.asarray(force_profile)
        if force_profile.ndim == sides_with_forces:
            force_profile = force_profile[
                :,
                np.newaxis,
                :,
                :,
            ]  # Ensure shape (N, 1, 3, 10)
        return force_profile

    force_profiles_1 = _preprocess_force(force_profiles_1)
    force_profiles_2 = _preprocess_force(force_profiles_2)

    # Non-zero force counts per sample
    nz1 = np.count_nonzero(force_profiles_1, axis=(2, 3)).flatten()
    nz2 = np.count_nonzero(force_profiles_2, axis=(2, 3)).flatten()

    # Force magnitudes (L2 norm across time axis)
    mag1 = np.linalg.norm(force_profiles_1, axis=-1).flatten()
    mag1 = mag1[mag1 != 0]
    mag2 = np.linalg.norm(force_profiles_2, axis=-1).flatten()
    mag2 = mag2[mag2 != 0]

    # Define shared bin edges
    nz_all = np.concatenate([nz1, nz2])
    mag_all = np.concatenate([mag1, mag2])
    nz_bins = np.histogram_bin_edges(nz_all, bins=30)
    mag_bins = np.histogram_bin_edges(mag_all, bins=30)

    # Consistent colors
    blue_force_visualization = "#1f77b4"
    orange_force_visualization = "#ff7f0e"

    # Plotting
    fig, axes_array = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    # Plot non-zero force counts
    axes_array[0].hist(
        nz1,
        bins=nz_bins,
        alpha=0.6,
        label=label_1,
        color=blue_force_visualization,
        edgecolor="black",
    )
    axes_array[0].hist(
        nz2,
        bins=nz_bins,
        alpha=0.6,
        label=label_2,
        color=orange_force_visualization,
        edgecolor="black",
    )
    axes_array[0].set_title("Non-zero Force Counts")
    axes_array[0].set_xlabel("Number of Non-zero Forces (per sample)")
    axes_array[0].set_ylabel("Sample Count")
    axes_array[0].legend()
    axes_array[0].grid(visible=True)

    # Plot force magnitude distribution
    axes_array[1].hist(
        mag1,
        bins=mag_bins,
        alpha=0.6,
        label=label_1,
        color=blue_force_visualization,
        edgecolor="black",
    )
    axes_array[1].hist(
        mag2,
        bins=mag_bins,
        alpha=0.6,
        label=label_2,
        color=orange_force_visualization,
        edgecolor="black",
    )
    axes_array[1].set_title("Force Magnitude Distribution")
    axes_array[1].set_xlabel("L2 Norm of Force")
    axes_array[1].set_ylabel("Frequency")
    axes_array[1].legend()
    axes_array[1].grid(visible=True)
    plt.show()

def analyse_force_generator() -> None:
    """Generate and visualise force profiles."""
    generator = ForceProfileGenerator(profile_top_and_sides=(10, 10), force_bounds=(0.1, 10.0), energy_bounds=(1e2, 5e4), batch_seed=42)
    force_profiles = generator.merger(num_samples=100_000)
    #force_profiles = generator.triangular_only(num_samples=100_000)
    force_profile_energy = np.sum(force_profiles**2, axis=(1, 2))

    # Find empty profiles
    empty_profiles = np.where(force_profile_energy == 0)[0]
    if len(empty_profiles) > 0:
        logger.warning(f"Found {len(empty_profiles)} empty force profiles at indices: {empty_profiles}")

    logger.info(f"Generated profiles like: {force_profiles[0]} and {force_profiles[1]}")
    max_force_value = np.max(np.abs(force_profiles))
    logger.info(f"Maximum force value across all profiles: {max_force_value}")

    visualise_profiles(force_profiles, force_profile_energy)

def analyse_raw_data() -> None:
    """Load and visualize forward model data."""
    data = forward_data_reader()
    if data is None:
        logger.error("Failed to load the forward model data.")
        return
    _, force_profiles, final_output_densities = data

    if force_profiles is None or final_output_densities is None:
        logger.error("Missing force or density data.")
        return

    triangular_data = forward_data_reader(Path(__file__).parent.parent.parent / Path("data", "raw", "triangular"))
    if triangular_data is None:
        logger.error("Failed to load the forward model data.")
        return
    _, triangular_force_profiles, triangular_final_output_densities = triangular_data

    if triangular_force_profiles is None or triangular_final_output_densities is None:
        logger.error("Missing triangular force or density data.")
        return

    force_mask = np.ones_like(force_profiles[0], dtype=bool)

    visualize_force_comparison(force_profiles, triangular_force_profiles)

    _, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i in range(4):
        random_index = np.random.randint(0, len(final_output_densities))
        plot_density_matrix(
            matrix=final_output_densities[random_index],
            force_data=(force_profiles[random_index], force_mask),
            axis=axes[i // 2, i % 2],
            title=f"Data at: {random_index}",
        )
    plt.show()


if __name__ == "__main__":
    analyse_force_generator()
    analyse_raw_data()
