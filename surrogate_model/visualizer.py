"""Visualize the surrogate model's predictions against actual validation data."""

import logging
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def plot_surrogate_model(
    predicted_matrices: np.ndarray,
    true_matrices: np.ndarray,
    sample_count: int = 3,
    show_plot: bool = True,
) -> List[plt.Figure]:
    """Compare the surrogate model's predictions with the actual validation data.

    Args:
    ----
        predicted_matrices (np.ndarray): Predicted density matrices from the surrogate model.
        true_matrices (np.ndarray): Actual density matrices from the validation set.
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
        raise ValueError("predicted_matrices and true_matrices must have the same shape.")
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

        # Plot original and predicted matrices side by side
        figure, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_density_matrix(
            actual_matrix,
            "Original Density Matrix",
            axes[0],
            min_true_value,
            max_true_value,
        )
        plot_density_matrix(
            predicted_matrix,
            "Predicted Density Matrix",
            axes[1],
            min_true_value,
            max_true_value,
        )

        figures.append(figure)

        # Adjust layout and show the plot
        plt.tight_layout()

        if show_plot:
            plt.show()

        logging.info("-" * 50)
    return figures


def plot_density_matrix(matrix: np.ndarray, title: str, ax, vmin, vmax) -> None:
    """Plot a density matrix with annotations.

    Args:
    ----
        matrix (np.ndarray): The density matrix to plot.
        title (str): Title of the plot.
        ax: Matplotlib axis to plot on.
        vmin (float): Minimum value for color scaling.
        vmax (float): Maximum value for color scaling.

    """
    ax.imshow(matrix, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color = "white",
                fontsize=8,
            )

def main() -> None:
    """Demonstrate the surrogate model visualization."""
    # Example usage with random data
    predicted_matrices = np.random.randn(5, 10, 10)  # Random predicted matrices
    true_matrices = np.random.randn(5, 10, 10)       # Random actual matrices
    plot_surrogate_model(
        predicted_matrices=predicted_matrices, true_matrices=true_matrices, sample_count=3
    )


if __name__ == "__main__":
    main()
