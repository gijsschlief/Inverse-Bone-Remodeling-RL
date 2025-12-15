"""Evaluate the inverse surrogate model's performance on validation data."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn.modules.module import Module

from bone_remodeling.src.forward_data.reader import forward_data_reader
from bone_remodeling.src.inverse_model.inverse_neural_network import (
    InverseModel,
)
from bone_remodeling.src.surrogate_model.evaluator import (
    average_similarity_score,
    validate_surrogate_model,
)
from bone_remodeling.src.surrogate_model.sanitizer import sanitize_data
from bone_remodeling.src.surrogate_model.splitter import splitting
from bone_remodeling.src.surrogate_model.visualizer import plot_surrogate_model

logger = logging.getLogger(__name__)


def run_inverse_model_evaluation(
    model_path: Path,
    data_path: Path,
    model_class: type[torch.nn.Module],
) -> None:
    """Load data, preprocess it, load the surrogate model, and evaluate its performance."""
    model = _load_model(model_path, model_class)
    if model is None:
        logger.error("Failed to load the inverse model.")
        return

    data = forward_data_reader(data_path)
    if data is None:
        logger.error("Failed to load the forward model data.")
        return
    _, force_profiles, final_output_densities = data

    if force_profiles is None or final_output_densities is None:
        logger.error("Failed to load the data.")
        return
    force_profiles, final_output_densities = sanitize_data(
        force_profiles,
        final_output_densities,
    )

    x_train, x_val, x_test, y_train, y_val, y_test = splitting(
        force_profiles,
        final_output_densities,
        random_state=1,
    )
    if x_val is None or y_val is None:
        logger.error("Failed to split the data into validation sets.")
        return

    predicted_matrices, true_matrices = validate_surrogate_model(model, x_val, y_val)
    if hasattr(predicted_matrices, "detach"):
        predicted_matrices = predicted_matrices.detach().cpu().numpy()
    if hasattr(true_matrices, "detach"):
        true_matrices = true_matrices.detach().cpu().numpy()

    plot_worst_prediction(true_matrices, predicted_matrices, x_val)

    average_similarity = average_similarity_score(
        predicted_matrices,
        true_matrices,
        baseline=0.1,
        threshold=0.5,
        method="ssim",
    )

    logger.info(f"The average similarity = {average_similarity}")

    plot_surrogate_model(
        predicted_matrices=predicted_matrices,
        true_matrices=true_matrices,
        force_profiles=x_val,
        sample_count=20,
        show_plot=True,
    )
    return

def load_inverse_model(
    model_path: Path,
    model_class: type[Module],
) -> Module | None:
    """Load the inverse surrogate model and its normalization parameters from a file.

    Args:
    ----
        model_path (Path): The path to the model file.
        model_class (type[Module]): The class of the model to be loaded.

    Returns:
    -------
        tuple: A tuple containing the model, input mean, input std, output mean, and output std.

    """
    try:
        checkpoint = torch.load(model_path)
        model = model_class()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None

def _load_model(
    model_path: Path,
    model_class: type[Module],
) -> Module | None:
    model = load_inverse_model(
        model_path,
        model_class,
    )
    if model is None:
        raise RuntimeError("Failed to load the surrogate model.")
    return model


def plot_worst_prediction(
    true_matrices: np.ndarray,
    predicted_matrices: np.ndarray,
    force_profiles: np.ndarray,
    count: int = 1,
) -> None:
    """Find the samples with the largest differences and plot it.

    Args:
    ----
        true_matrices (np.ndarray): The true matrices of shape (N, 10, 10).
        predicted_matrices (np.ndarray): The predicted matrices of shape (N, 10, 10).
        force_profiles (np.ndarray): The force profiles of shape (N, 3, 10).
        count (int): The number of samples to plot with the largest differences.

    """
    offset = predicted_matrices - true_matrices
    largest_differences = np.abs(offset).mean(axis=(1, 2)).argsort()[::-1]

    logger.info(f"Largest differences in predicted matrices: {largest_differences}")

    bad_prediction = predicted_matrices[largest_differences[:count]]
    bad_originals = true_matrices[largest_differences[:count]]
    bad_forces = force_profiles[largest_differences[:count]]
    plot_surrogate_model(
        predicted_matrices=bad_prediction,
        true_matrices=bad_originals,
        force_profiles=bad_forces,
        sample_count=count,
        show_plot=True,
    )


if __name__ == "__main__":
    model_path = Path("/home/gijs/Desktop/Thesis/data/inverse_model/trained_model.pth")
    data_path = Path("/home/gijs/Desktop/Thesis/data/raw/triangular/")
    run_inverse_model_evaluation(
        model_path=model_path,
        data_path=data_path,
        model_class=InverseModel,
    )
