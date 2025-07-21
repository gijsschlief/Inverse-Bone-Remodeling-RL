"""Visualize the surrogate model's predictions against actual validation data."""

import logging
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from bone_remodeling.src.forward_model.density_visualizer import (
    plot_density_matrix,  # type: ignore
)
from bone_remodeling.src.forward_model.parameters import SimulationParameters

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def plot_surrogate_model(
    predicted_matrices: np.ndarray,
    true_matrices: np.ndarray,
    force_profiles: np.ndarray | None = None,
    sample_count: int = 3,
    show_plot: bool = True,
) -> List[plt.Figure]:
    """Compare the surrogate model's predictions with the actual validation data.

    Args:
    ----
        predicted_matrices (np.ndarray): Predicted density matrices from the surrogate model.
        true_matrices (np.ndarray): Actual density matrices from the validation set.
        force_profiles (np.ndarray, None): Force profiles corresponding to the matrices.
        sample_count (int): Number of random samples to visualize. Default is 3.
        show_plot (bool): Whether to display the plots. Default is True.

    """
    if not isinstance(predicted_matrices, np.ndarray) or not isinstance(
        true_matrices, np.ndarray
    ):
        raise ValueError(
            "Both predicted_matrices and true_matrices must be numpy.ndarray objects."
        )
    if predicted_matrices.shape != true_matrices.shape:
        raise ValueError(
            "predicted_matrices and true_matrices must have the same shape."
        )
    if sample_count <= 0 or sample_count > len(true_matrices):
        raise ValueError(
            "sample_count must be a positive integer less than or equal to the number of validation samples."
        )

    random_indices = random.sample(range(len(true_matrices)), sample_count)

    max_true_value = np.max(true_matrices)
    min_true_value = np.min(true_matrices)

    logging.info("=== Surrogate Model Comparison ===")
    figures = []
    for idx in random_indices:
        predicted_matrix = predicted_matrices[idx]
        actual_matrix = true_matrices[idx]

        logging.info(f"Sample Index: {idx}\nOriginal Density Matrix:\n{actual_matrix}")
        logging.info(f"Predicted Density Matrix:\n{predicted_matrix}")
        logging.info(
            f"Force Profile: {force_profiles[idx] if force_profiles is not None else 'N/A'}"
        )

        # Plot original, predicted, and difference matrices side by side (1 row, 3 columns)
        figure, axes = plt.subplots(1, 3, figsize=(18, 6))
        plot_density_matrix(
            actual_matrix,
            force_profiles[idx] if force_profiles is not None else None,
            "Original Density Matrix",
            axes[0],
            min_true_value,
            max_true_value,
        )
        plot_density_matrix(
            predicted_matrix,
            force_profiles[idx] if force_profiles is not None else None,
            "Predicted Density Matrix",
            axes[1],
            min_true_value,
            max_true_value,
        )
        plot_difference_matrix(
            predicted_matrix,
            actual_matrix,
            "Difference Matrix (Predicted - Actual) as Percentage",
            axes[2],
            color_scale_min=-100,
            color_scale_max=100,
        )

        figures.append(figure)
        plt.tight_layout()

        if show_plot:
            plt.show()

        logging.info("-" * 50)
    return figures


def plot_difference_matrix(
    predicted_matrix: np.ndarray,
    actual_matrix: np.ndarray,
    title: str,
    axis: plt.Axes,
    color_scale_min: float = -100,
    color_scale_max: float = 100,
    color_bar: bool = True,
) -> None:
    """Plot the difference between predicted and actual matrices as percentage with a diverging colormap.

    Args:
    ----
        predicted_matrix (np.ndarray): Predicted density matrix.
        actual_matrix (np.ndarray): Actual density matrix.
        title (str): Title of the plot.
        axis: Matplotlib axis to plot on.
        color_scale_min (float): Minimum value for color scaling. Default is -100
        color_scale_max (float): Maximum value for color scaling. Default is 100
        color_bar (bool): Whether to include a color bar. Default is True.

    """
    difference_matrix = np.divide(predicted_matrix - actual_matrix, np.ones_like(actual_matrix)*SimulationParameters.max_density) * 100
    # Use a diverging colormap: blue (under), white (exact), red (over)
    difference_image = axis.imshow(
        difference_matrix,
        cmap="seismic",
        interpolation="nearest",
        vmin=color_scale_min,
        vmax=color_scale_max,
    )
    axis.set_title(title)
    axis.set_xlabel("Columns")
    axis.set_ylabel("Rows")
    for i in range(difference_matrix.shape[0]):
        for j in range(difference_matrix.shape[1]):
            axis.text(
                j,
                i,
                f"{difference_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=(
                    "black"
                    if abs(difference_matrix[i, j])
                    < (color_scale_max - color_scale_min) / 4
                    else "white"
                ),
                fontsize=8,
            )
    if color_bar:
        plt.colorbar(difference_image, ax=axis, fraction=0.046, pad=0.04)


def main() -> None:
    """Demonstrate the surrogate model visualization using example data."""
    predicted_matrices = np.random.randn(5, 10, 10)
    true_matrices = np.random.randn(5, 10, 10)
    plot_surrogate_model(
        predicted_matrices=predicted_matrices,
        true_matrices=true_matrices,
        force_profiles=np.random.randn(5, 3, 10),
        sample_count=1,
    )

    # TEST IF THE RIGHT WAY IS UP
    gradient_example = np.arange(100).reshape(10, 10)
    plot_density_matrix(
        gradient_example / 100,
        None,
        "Test Matrix",
        plt.gca(),
        color_scale_min=0,
        color_scale_max=1,
    )
    plt.show()


if __name__ == "__main__":
    main()
