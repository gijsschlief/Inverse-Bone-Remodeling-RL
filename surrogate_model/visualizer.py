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
            "Difference Matrix (Predicted - Actual)",
            axes[2],
            color_scale_min=-1.0,
            color_scale_max=1.0,
        )

        figures.append(figure)
        plt.tight_layout()

        if show_plot:
            plt.show()

        logging.info("-" * 50)
    return figures


def plot_density_matrix(
    matrix: np.ndarray,
    force_profile: np.ndarray | None,
    title: str,
    axis: plt.Axes,
    color_scale_min: float = 0.01,
    color_scale_max: float = 1.73,
) -> None:
    """Plot a density matrix with annotations.

    Args:
    ----
        matrix (np.ndarray): Density matrix to plot.
        force_profile (np.ndarray | None): Force profile corresponding to the matrix.
        title (str): Title of the plot.
        axis: Matplotlib axis to plot on.
        color_scale_min (float): Minimum value for color scaling.
        color_scale_max (float): Maximum value for color scaling.

    """
    axis.imshow(
        matrix,
        cmap="viridis",
        interpolation="nearest",
        vmin=color_scale_min,
        vmax=color_scale_max,
    )
    axis.set_title(title)
    axis.set_xlabel("Columns")
    axis.set_ylabel("Rows")

    # Annotate matrix values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    def get_force_color(
        force_val: float, min_force: float = 0.1, max_force: float = 10.0
    ) -> str:
        """Map a force magnitude to a grayscale hex color between light gray and black."""
        abs_force = abs(force_val)

        if abs_force < min_force:
            return "#000000"  # Light gray for very small forces
        if abs_force > max_force:
            return "#FF0000"  # Black for very large forces

        # Normalize force value between 0 and 1
        norm = min(max((abs_force - min_force) / (max_force - min_force), 0.0), 1.0)

        # Interpolate between black (0,0,0) and red (255,0,0)
        red = int(255 * norm)
        green = 0
        blue = 0

        hex_color = f"#{red:02x}{green:02x}{blue:02x}"
        return hex_color

    # Plot force arrows if force_profile is provided
    if force_profile is not None:
        height, width = matrix.shape
        top_forces = force_profile[0]
        right_forces = force_profile[1]
        left_forces = force_profile[2]

        # Compute max magnitude for scaling (avoid division by zero)
        max_force = np.max(np.abs(force_profile))

        arrow_scale = 0.5  # Maximum arrow length

        # Top forces: draw downward arrows above row 0
        for j in range(width):
            if abs(top_forces[j]) > 1e-3:
                scaled_length = arrow_scale * abs(top_forces[j]) / max_force
                force_color = get_force_color(top_forces[j])
                if top_forces[j] < 0:
                    axis.arrow(
                        j,
                        -0.5,
                        0,
                        scaled_length * np.sign(top_forces[j]),  # downward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )
                elif top_forces[j] > 0:
                    axis.arrow(
                        j,
                        -0.5 - scaled_length - 0.15,
                        0,
                        scaled_length * np.sign(top_forces[j]),  # upward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )

        # Left forces: draw rightward arrows left of column 0
        for i in range(height):
            if abs(left_forces[i]) > 1e-3:
                scaled_length = arrow_scale * abs(left_forces[i]) / max_force
                force_color = get_force_color(left_forces[i])
                if left_forces[i] < 0:
                    axis.arrow(
                        -0.5,
                        height - 1 - i,
                        scaled_length * np.sign(left_forces[i]),
                        0,  # rightward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )
                elif left_forces[i] > 0:
                    axis.arrow(
                        -0.5 - scaled_length - 0.15,
                        height - 1 - i,
                        scaled_length * np.sign(left_forces[i]),
                        0,  # leftward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )

        # Right forces: draw leftward arrows right of last column
        for i in range(height):
            if abs(right_forces[i]) > 1e-3:
                scaled_length = arrow_scale * abs(right_forces[i]) / max_force
                force_color = get_force_color(right_forces[i])
                if right_forces[i] < 0:
                    axis.arrow(
                        width - 0.5 + scaled_length + 0.15,
                        height - 1 - i,
                        scaled_length * np.sign(right_forces[i]),
                        0,  # rightward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )
                elif right_forces[i] > 0:
                    axis.arrow(
                        width - 0.5,
                        height - 1 - i,
                        scaled_length * np.sign(right_forces[i]),
                        0,  # leftward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )


def plot_difference_matrix(
    predicted_matrix: np.ndarray,
    actual_matrix: np.ndarray,
    title: str,
    axis: plt.Axes,
    color_scale_min: float = -1.0,
    color_scale_max: float = 1.0,
) -> None:
    """Plot the difference between predicted and actual matrices with a diverging colormap.

    Args:
    ----
        predicted_matrix (np.ndarray): Predicted density matrix.
        actual_matrix (np.ndarray): Actual density matrix.
        title (str): Title of the plot.
        axis: Matplotlib axis to plot on.
        color_scale_min (float): Minimum value for color scaling. Default is -1.0
        color_scale_max (float): Maximum value for color scaling. Default is 1.0

    """
    difference_matrix = predicted_matrix - actual_matrix
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
    plt.colorbar(difference_image, ax=axis, fraction=0.046, pad=0.04)


def main() -> None:
    """Demonstrate the surrogate model visualization using example data."""
    predicted_matrices = np.random.randn(5, 10, 10)
    true_matrices = np.random.randn(5, 10, 10)
    plot_surrogate_model(
        predicted_matrices=predicted_matrices,
        true_matrices=true_matrices,
        force_profiles=np.random.randn(5, 3, 10),
        sample_count=3,
    )


if __name__ == "__main__":
    main()
