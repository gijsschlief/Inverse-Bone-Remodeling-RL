"""Evaluate the surrogate model's performance on validation data."""

import logging
from typing import Tuple

import numpy as np
from bone_remodeling.forward_model.data_reader import forward_data_reader
from bone_remodeling.surrogate_model.neural_networks.large_nn import (
    LargeSurrogateModel,
)
from bone_remodeling.surrogate_model.neural_networks.medium_nn import (
    MediumSurrogateModel,
)

from surrogate_model.evaluator import average_similarity_score, validate_surrogate_model
from surrogate_model.loader import load_surrogate_model
from surrogate_model.splitter import splitting
from surrogate_model.visualizer import plot_surrogate_model


def sanitize_matrices(
    predicted_matrices: np.ndarray, true_matrices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Sanitize the predicted and true matrices by filtering out any entries that contain NaN values."""
    if not isinstance(predicted_matrices, np.ndarray) or not isinstance(
        true_matrices, np.ndarray
    ):
        logging.error(
            "Both predicted_matrices and true_matrices must be lists of numpy arrays."
        )
        raise ValueError(
            "Both predicted_matrices and true_matrices must be lists of numpy arrays."
        )

    filtered_predicted = []
    filtered_true = []
    nan_count = 0

    for pred, true in zip(predicted_matrices, true_matrices):
        if not np.isnan(pred).any() and not np.isnan(true).any():
            filtered_predicted.append(pred)
            filtered_true.append(true)
        else:
            nan_count += 1

    if nan_count > 0:
        logging.error(
            f"Found {nan_count} entries with NaN values in predicted or true matrices. These entries were filtered out."
        )

    return np.array(filtered_predicted), np.array(filtered_true)


def main() -> None:
    """Load data, preprocess it, load the surrogate model, and evaluate its performance."""
    model = load_surrogate_model(
        "/home/gijs/Desktop/Thesis/data/models/trained_model_8.pth",
        MediumSurrogateModel,
    )
    if model is None:
        logging.error("Failed to load the surrogate model.")
        return

    data = forward_data_reader("/home/gijs/Desktop/Thesis/data/raw/")
    if data is None:
        logging.error("Failed to load the forward model data.")
        return
    _, force_profiles, final_output_densities = data

    if force_profiles is None or final_output_densities is None:
        logging.error("Failed to load the data.")
        return
    _, X_val, _, _, y_val, _ = splitting(
        force_profiles, final_output_densities, random_state=0
    )
    if X_val is None or y_val is None:
        logging.error("Failed to split the data into validation sets.")
        return

    predicted_matrices, true_matrices = validate_surrogate_model(model, X_val, y_val)

    predicted_matrices, true_matrices = sanitize_matrices(
        predicted_matrices, true_matrices
    )

    average_similarity = average_similarity_score(
        predicted_matrices, true_matrices, baseline=0.1, threshold=0.5, method="ssim"
    )

    logging.info(f"The average similarity = {average_similarity}")

    plot_surrogate_model(
        predicted_matrices=predicted_matrices,
        true_matrices=true_matrices,
        sample_count=3,
        show_plot=True,
    )
    return


if __name__ == "__main__":
    main()
