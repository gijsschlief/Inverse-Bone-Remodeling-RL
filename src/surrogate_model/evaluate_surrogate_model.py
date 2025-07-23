"""Evaluate the surrogate model's performance on validation data."""

import logging
from pathlib import Path

import numpy as np
from torch.nn.modules.module import Module

from bone_remodeling.src.forward_data.reader import forward_data_reader
from bone_remodeling.src.surrogate_model.evaluator import (
    average_similarity_score,
    validate_surrogate_model,
)
from bone_remodeling.src.surrogate_model.loader import load_surrogate_model
from bone_remodeling.src.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)
from bone_remodeling.src.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.src.surrogate_model.normalizor import (
    normalize_data,
    unnormalize_data,
)
from bone_remodeling.src.surrogate_model.sanitizer import sanitize_data
from bone_remodeling.src.surrogate_model.splitter import splitting
from bone_remodeling.src.surrogate_model.visualizer import plot_surrogate_model

logger = logging.getLogger(__name__)

def run_model_evaluation(model_path: Path, data_path: Path, model_class: type[SurrogateModel]) -> None:
    """Load data, preprocess it, load the surrogate model, and evaluate its performance."""
    model, x_mean, x_std, y_mean, y_std = _load_model(model_path, model_class)

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
        random_state=0,
    )
    if x_val is None or y_val is None:
        logger.error("Failed to split the data into validation sets.")
        return

    # Normalize the validation data if normalization parameters are available
    if x_mean is not None and x_std is not None:
        x_val_unnormalized = x_val.copy()
        x_train, _, _ = normalize_data(x_train, x_mean, x_std)
        x_val, _, _ = normalize_data(x_val, x_mean, x_std)
        x_test, _, _ = normalize_data(x_test, x_mean, x_std)

    if y_mean is not None and y_std is not None:
        y_train, _, _ = normalize_data(y_train, y_mean, y_std)
        y_val, _, _ = normalize_data(y_val, y_mean, y_std)
        y_test, _, _ = normalize_data(y_test, y_mean, y_std)
        output_normalized = True
    else:
        output_normalized = False

    predicted_matrices, true_matrices = validate_surrogate_model(model, x_val, y_val)

    if output_normalized and y_mean is not None and y_std is not None:
        predicted_matrices = unnormalize_data(predicted_matrices, y_mean, y_std)
        true_matrices = unnormalize_data(true_matrices, y_mean, y_std)
        if hasattr(predicted_matrices, "detach"):
            predicted_matrices = predicted_matrices.detach().cpu().numpy()
        if hasattr(true_matrices, "detach"):
            true_matrices = true_matrices.detach().cpu().numpy()

    plot_worst_prediction(true_matrices, predicted_matrices, x_val_unnormalized)

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
        force_profiles=x_val_unnormalized,
        sample_count=20,
        show_plot=True,
    )
    return

def _load_model(model_path: Path, model_class: type[Module]) -> tuple[Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_and_normalization_params = load_surrogate_model(
        model_path,
        model_class,
    )
    if model_and_normalization_params is None:
        raise RuntimeError("Failed to load the surrogate model and normalization parameters.")
    model, x_mean, x_std, y_mean, y_std = model_and_normalization_params
    if model is None:
        raise RuntimeError("Failed to load the surrogate model.")
    if x_mean is None or x_std is None or y_mean is None or y_std is None:
        raise RuntimeError("Normalization parameters are missing.")
    return model, x_mean, x_std, y_mean, y_std

def plot_worst_prediction(true_matrices: np.ndarray, predicted_matrices: np.ndarray, force_profiles: np.ndarray, count: int = 1) -> None:
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
    model_path = Path("/home/gijs/Desktop/Thesis/data/models/trained_model_4.pth")
    data_path = Path("/home/gijs/Desktop/Thesis/data/raw/")
    run_model_evaluation(model_path=model_path, data_path=data_path, model_class=ReversedSurrogateModel)
