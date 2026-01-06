"""Evaluate an ensemble of surrogate models and visualize their predictions."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bone_remodeling.src.forward_data.visualizer import plot_density_matrix
from bone_remodeling.src.surrogate_model.ensemble import (
    load_ensemble_models,
    predict_with_ensemble,
)
from bone_remodeling.src.surrogate_model.evaluator import average_similarity_score
from bone_remodeling.src.surrogate_model.loader import SurrogateModelLoader
from bone_remodeling.src.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)
from bone_remodeling.src.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.src.surrogate_model.trainer import load_and_split_data
from bone_remodeling.src.surrogate_model.visualizer import plot_difference_matrix

logger = logging.getLogger(__name__)


def plot_ensemble(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    true_matrices: np.ndarray,
    force_profiles: np.ndarray,
) -> None:
    """Plot the ensemble predictions against the true values.

    Args:
    ----
        mean_pred (np.ndarray): The mean predictions from the ensemble.
        std_pred (np.ndarray): The standard deviation of the predictions from the ensemble.
        true_matrices (np.ndarray): The true values to compare against.
        force_profiles (np.ndarray): The force profiles used for prediction.

    """
    axes = plt.subplots(2, 2, figsize=(12, 12))[1]
    plot_density_matrix(
        matrix=mean_pred,
        force_profile=force_profiles,
        title="Ensemble Mean Prediction",
        axis=axes[0, 0],
    )
    plot_density_matrix(
        matrix=true_matrices,
        force_profile=force_profiles,
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
        force_profile=force_profiles,
        title="Prediction Uncertainty (Std Dev)",
        axis=axes[1, 1],
        color_scale=(0, 0.5),
    )
    plt.tight_layout()
    plt.show()


def main(
    model_paths: list[Path],
    model_class: type[SurrogateModel],
    model_loader: type[SurrogateModelLoader],
    data_file_path: Path,
    random_state: int = 0,
) -> None:
    """Load models, make predictions, and save results."""
    # Load data
    (
        x_train_np,
        x_val_np,
        x_test_np,
        y_train_np,
        y_val_np,
        y_test_np,
    ) = load_and_split_data(data_file_path, random_state=random_state)

    # Load models
    models, x_means, x_stds, y_means, y_stds = load_ensemble_models(
        model_paths,
        model_class,
        model_loader,
    )

    # Make predictions on test set
    x_normalizations = (x_means, x_stds)
    y_normalizations = (y_means, y_stds)

    for _ in range(100):
        k = np.random.randint(0, len(x_test_np))
        mean_pred, std_pred = predict_with_ensemble(
            models,
            x_test_np[k],
            x_normalizations,
            y_normalizations,
        )
        sample_similarity = average_similarity_score(
        mean_pred.squeeze(),
        y_test_np[k],
        baseline=0.1,
        threshold=0.5,
        method="ssim",
        )
        logger.info(f"Sample {k} similarity = {sample_similarity}")
        plot_ensemble(
            mean_pred.squeeze(),
            std_pred.squeeze(),
            y_test_np[k],
            x_test_np[k],
        )

    # Calculate average similarity score
    mean_preds, std_preds = predict_with_ensemble(
        models,
        x_test_np,
        x_normalizations,
        y_normalizations,
    )

    average_similarity = average_similarity_score(
        mean_preds,
        y_test_np,
        baseline=0.1,
        threshold=0.5,
        method="ssim",
    )

    logger.info(f"The average similarity = {average_similarity}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_paths = [
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_new_data_1.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_new_data_2.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_new_data_3.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_new_data_4.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_new_data_5.pth"),
    ]
    data_file_path = Path("/home/gijs/Desktop/Thesis/data/raw/")

    main(
        model_paths,
        model_class=ReversedSurrogateModel,
        model_loader=SurrogateModelLoader,
        data_file_path=data_file_path,
        random_state=1,
    )
