"""Visualization of Forward Model Data."""

import logging

import matplotlib.pyplot as plt
import numpy as np

from bone_remodeling.src.forward_data.reader import forward_data_reader  # type: ignore
from bone_remodeling.src.forward_model.density_visualizer import (
    plot_density_matrix,  # type: ignore
)

logger = logging.getLogger(__name__)


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
    color1 = "#1f77b4"  # blue
    color2 = "#ff7f0e"  # orange

    # Plotting
    fig, axes_array = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    # Plot non-zero force counts
    axes_array[0].hist(
        nz1,
        bins=nz_bins,
        alpha=0.6,
        label=label_1,
        color=color1,
        edgecolor="black",
    )
    axes_array[0].hist(
        nz2,
        bins=nz_bins,
        alpha=0.6,
        label=label_2,
        color=color2,
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
        color=color1,
        edgecolor="black",
    )
    axes_array[1].hist(
        mag2,
        bins=mag_bins,
        alpha=0.6,
        label=label_2,
        color=color2,
        edgecolor="black",
    )
    axes_array[1].set_title("Force Magnitude Distribution")
    axes_array[1].set_xlabel("L2 Norm of Force")
    axes_array[1].set_ylabel("Frequency")
    axes_array[1].legend()
    axes_array[1].grid(visible=True)

    plt.show()


def main() -> None:
    """Load and visualize forward model data."""
    data = forward_data_reader(
        file_path="/home/gijs/Desktop/Thesis/data/raw/training_combined_third_order_100_samples_0717_1909.json",
    )
    if data is None:
        logger.error("Failed to load the forward model data.")
        return

    _, force_profiles, final_output_densities = data

    if force_profiles is None or final_output_densities is None:
        logger.error("Missing force or density data.")
        return

    old_data = forward_data_reader(
        file_path="/home/gijs/Desktop/Thesis/data/raw/training_combined_third_order_30000_samples_0718_0652.json",
    )
    if old_data is None:
        logger.error("Failed to load the forward model data.")
        return

    _, old_force_profiles, old_final_output_densities = old_data

    if old_force_profiles is None or old_final_output_densities is None:
        logger.error("Missing force or density data.")
        return

    visualize_force_comparison(force_profiles, old_force_profiles)

    _, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i in range(4):
        random_index = np.random.randint(0, len(old_final_output_densities))
        plot_density_matrix(
            old_final_output_densities[random_index],
            force_profile=old_force_profiles[random_index],
            axis=axes[i // 2, i % 2],
            title=f"Data at: {random_index}",
        )
    plt.show()


if __name__ == "__main__":
    main()
