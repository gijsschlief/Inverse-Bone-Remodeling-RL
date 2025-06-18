import logging
import random
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def plot_surrogate_model(
    val_logits: torch.Tensor,
    y_val_tensor: torch.Tensor,
    sample_count: int = 3,
    show_plot: bool = True,
) -> List[plt.Figure]:
    """
    Compare the surrogate model's predictions with the actual validation data.
    Args:
        val_logits (torch.Tensor): Predicted density matrices from the surrogate model.
        y_val_tensor (torch.Tensor): Actual density matrices from the validation set.
        sample_count (int): Number of random samples to visualize. Default is 3.
    """
    if not isinstance(val_logits, torch.Tensor) or not isinstance(
        y_val_tensor, torch.Tensor
    ):
        raise ValueError(
            "Both val_logits and y_val_tensor must be torch.Tensor objects."
        )
    if val_logits.shape != y_val_tensor.shape:
        raise ValueError("val_logits and y_val_tensor must have the same shape.")
    if sample_count <= 0 or sample_count > len(y_val_tensor):
        raise ValueError(
            "sample_count must be a positive integer less than or equal to the number of validation samples."
        )

    random_indices = random.sample(range(len(y_val_tensor)), sample_count)

    max_true_value = torch.max(y_val_tensor).item()
    min_true_value = torch.min(y_val_tensor).item()

    logging.info("=== Surrogate Model Comparison ===")
    figures = []
    for idx in random_indices:
        predicted_matrix = val_logits[idx].cpu().numpy()
        actual_matrix = y_val_tensor[idx].cpu().numpy()

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


def plot_density_matrix(matrix: np.ndarray, title: str, ax, vmin, vmax):
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
                color="white",
                fontsize=8,
            )


if __name__ == "__main__":
    # Example val_logits and y_val_tensor for demonstration purposes
    val_logits = torch.randn(5, 10, 10)
    y_val_tensor = torch.randn(5, 10, 10)
    plot_surrogate_model(
        val_logits=val_logits, y_val_tensor=y_val_tensor, sample_count=1
    )
