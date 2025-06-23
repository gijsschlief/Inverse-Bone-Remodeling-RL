"""Visualization of Forward Model Data."""

import logging

import matplotlib.pyplot as plt
import numpy as np

from forward_model.data_reader import forward_data_reader

logging.basicConfig(level=logging.INFO)

def visualize_data(force_profiles: np.ndarray, output_densities: np.ndarray, title: str = "Data Visualization") -> None:
    """Visualizes the given data using matplotlib.

    Args:
    ----
        force_profiles (np.ndarray): Shape (samples, 1, 3, 10)
        output_densities (np.ndarray): Shape (samples, 1, 10, 10)
        title (str): Figure title

    """
    num_samples = min(3, force_profiles.shape[0])  # Limit for display

    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 5 * num_samples), constrained_layout=True)

    if num_samples == 1:
        axes = axes[np.newaxis, :]  # Ensure axes is always 2D

    for i in range(num_samples):
        # --- Force Profiles ---
        ax = axes[i, 0]
        force = np.squeeze(force_profiles[i])  # (3, 10)
        for j in range(force.shape[0]):
            ax.plot(force[j], label=f'Force {j+1}')
        ax.set_title(f"Sample {i+1} - Force Profiles")
        ax.set_xlabel("Time")
        ax.set_ylabel("Force Value")
        ax.legend(loc="upper right")
        ax.grid(True)

        # --- Output Density Heatmap ---
        ax = axes[i, 1]
        density = np.squeeze(output_densities[i])  # (10, 10)
        im = ax.imshow(density, cmap='viridis', aspect='equal')
        ax.set_title(f"Sample {i+1} - Output Density")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Density")

        # --- Density Histogram ---
        ax = axes[i, 2]
        ax.hist(density.flatten(), bins=20, color='orange', edgecolor='black')
        ax.set_title(f"Sample {i+1} - Density Histogram")
        ax.set_xlabel("Density Value")
        ax.set_ylabel("Count")
        ax.grid(True)

    fig.suptitle(title, fontsize=16)
    plt.show()


def main() -> None:
    """Load and visualize forward model data."""
    data = forward_data_reader(file_path="/home/gijs/Desktop/Thesis/data/raw/")
    if data is None:
        logging.error("Failed to load the forward model data.")
        return

    _, force_profiles, final_output_densities = data

    if force_profiles is None or final_output_densities is None:
        logging.error("Missing force or density data.")
        return

    visualize_data(force_profiles, final_output_densities, title="Forward Model Data Visualization")


if __name__ == "__main__":
    main()
