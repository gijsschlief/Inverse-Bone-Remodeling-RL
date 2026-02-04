"""Visualize the surrogate model's predictions against actual validation data."""

import logging
from dataclasses import dataclass

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from bone_remodelling.forward_model.density_visualizer import (
    plot_density_matrix,  # type: ignore
)
from bone_remodelling.forward_model.parameters import SimulationParameters

logger = logging.getLogger(__name__)

def plot_loss(train_losses: list[float], val_losses: list[float], learning_rates: list[float]) -> None:
    """Plot the training and validation loss history with learning rate overlay."""
    if not train_losses or not val_losses or not learning_rates:
        logger.error("One or more training history lists are empty.")
        return

    # Build and clear the plot
    fig = plt.figure(num=1, clear=True)
    axes = fig.add_subplot(1, 1, 1)

    # Plot Losses on the primary Y-axis
    line1, = axes.plot(train_losses, label="Train Loss", color="tab:blue")
    line2, = axes.plot(val_losses, label="Validation Loss", color="tab:orange")
    axes.set_yscale("log")
    axes.set_ylabel("Loss")
    axes.grid(visible=True, which="both", linestyle="--", alpha=0.3)

    # Create secondary Y-axis for Learning Rate
    ax2 = axes.twinx()
    line3, = ax2.plot(learning_rates, color="gray", linestyle=":", label="Learning Rate", alpha=0.7)
    ax2.set_yscale("log")
    ax2.set_ylabel("Learning Rate", color="gray")
    ax2.grid(visible=False)

    # Combine legends from both axes
    lines = [line1, line2, line3]
    legend_labels = [str(line.get_label()) for line in lines]
    axes.legend(lines, legend_labels, loc="upper right")
    axes.set_title("Training Progress: Loss & Learning Rate", fontsize=12, fontweight='bold')
    axes.set_xlabel("Epoch")
    plt.tight_layout()
    plt.show(block=False)

@dataclass
class PlottingParameters:
    """Parameters for plotting density matrices."""

    color_scale_min: float = -100
    color_scale_max: float = 100
    color_bar: bool = False

def plot_difference_matrix(
    predicted_matrix: np.ndarray,
    actual_matrix: np.ndarray,
    title: str,
    axis: matplotlib.axes.Axes,
    plotting_parameters: PlottingParameters = PlottingParameters(),
) -> None:
    """Plot the difference between predicted and actual matrices as percentage with a diverging colormap.

    Args:
    ----
        predicted_matrix (np.ndarray): Predicted density matrix.
        actual_matrix (np.ndarray): Actual density matrix.
        title (str): Title of the plot.
        axis: Matplotlib axis to plot on.
        plotting_parameters (PlottingParameters): Parameters for plotting, including color scale and color bar.

    """
    difference_matrix = (
        np.divide(
            predicted_matrix - actual_matrix,
            np.ones_like(actual_matrix) * SimulationParameters.max_density,
        )
        * 100
    )
    difference_image = axis.imshow(
        difference_matrix,
        cmap="seismic",
        interpolation="nearest",
        vmin=plotting_parameters.color_scale_min,
        vmax=plotting_parameters.color_scale_max,
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
                    < (
                        plotting_parameters.color_scale_max
                        - plotting_parameters.color_scale_min
                    )
                    / 4
                    else "white"
                ),
                fontsize=8,
            )
    if plotting_parameters.color_bar:
        plt.colorbar(difference_image, ax=axis, fraction=0.046, pad=0.04)

def plot_surrogate(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    true_matrices: np.ndarray,
    force_profiles: np.ndarray,
) -> None:
    """Plot the surrogate model predictions against the true values.

    Args:
    ----
        mean_pred (np.ndarray): The mean predictions from the surrogate model.
        std_pred (np.ndarray): The standard deviation of the predictions from the surrogate model.
        true_matrices (np.ndarray): The true values to compare against.
        force_profiles (np.ndarray): The force profiles used for prediction.

    """
    force_mask = np.ones_like(force_profiles, dtype=bool)
    axes = plt.subplots(2, 2, figsize=(12, 12))[1]
    plot_density_matrix(
        matrix=mean_pred,
        force_data=(force_profiles, force_mask),
        title="Surrogate Model Mean Prediction",
        axis=axes[0, 0],
    )
    plot_density_matrix(
        matrix=true_matrices,
        force_data=(force_profiles, force_mask),
        title="True Density",
        axis=axes[0, 1],
    )
    plot_difference_matrix(
        predicted_matrix=mean_pred,
        actual_matrix=true_matrices,
        title="Difference",
        axis=axes[1, 0],
    )
    plot_density_matrix(
        matrix=std_pred,
        force_data=(force_profiles, force_mask),
        title="Prediction Uncertainty (Std Dev)",
        axis=axes[1, 1],
        color_scale=(0, 0.5),
    )
    plt.tight_layout()
    plt.show()

def demonstrate_visualization() -> None:
    """Demonstrate the surrogate model visualization using example data."""
    predicted_matrices = np.random.randn(10, 10)
    true_matrices = np.random.randn(10, 10)
    plot_surrogate(
        mean_pred=predicted_matrices,
        std_pred=np.zeros_like(predicted_matrices),
        true_matrices=true_matrices,
        force_profiles=np.random.randn(3, 10),
    )

    # TEST IF THE RIGHT WAY IS UP
    gradient_example = np.arange(100).reshape(10, 10)
    ax = plt.subplots(figsize=(5, 5))[1]
    plot_density_matrix(
        gradient_example / 100,
        (None, None),
        "Test Matrix",
        ax,
        color_scale = (0, 1),
    )
    plt.show()

if __name__ == "__main__":
    demonstrate_visualization()
