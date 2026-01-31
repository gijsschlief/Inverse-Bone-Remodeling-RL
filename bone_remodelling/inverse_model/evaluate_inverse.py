"""Evaluate the inverse surrogate model's performance on validation data."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bone_remodelling.forward_data.reader import forward_data_reader
from bone_remodelling.forward_model.density_visualizer import plot_density_matrix
from bone_remodelling.inverse_model.inverse_neural_network import (
    InverseModel,
)
from bone_remodelling.inverse_model.inverse_trainer import params_to_force_profile
from bone_remodelling.rl_model.forward_pass import SurrogateForwarder
from bone_remodelling.rl_model.reward_calculation import calculate_similarity
from bone_remodelling.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodelling.surrogate_model.sanitizer import sanitize_data
from bone_remodelling.surrogate_model.splitter import splitting
from bone_remodelling.surrogate_model.visualizer import plot_difference_matrix

logger = logging.getLogger(__name__)

def inverse_model_metrics(sample_forces: np.ndarray, predicted_forces: np.ndarray) -> None:
    """Calculate metrics for the inverse model predictions. Looks at the force side, exact location and magnitude differences."""
    # Force Side Selection
    test_side = np.argmax(np.max(np.abs(sample_forces), axis=2), axis=1)
    eval_side = np.argmax(np.max(np.abs(predicted_forces), axis=2), axis=1)
    side_accuracy = np.sum(test_side == eval_side) / len(test_side)
    logger.info(f"Force Side Selection Accuracy: {np.sum(test_side == eval_side)}/{len(test_side)} or {side_accuracy:.4f} correct.")

    # Correct per-sample peak location extraction
    test_peak_locations = np.array([
        np.argmax(np.abs(sample_forces[i, test_side[i], :]))
        for i in range(len(sample_forces))
    ])

    eval_peak_locations_on_true_side = np.array([
        np.argmax(np.abs(predicted_forces[i, test_side[i], :]))
        for i in range(len(predicted_forces))
    ])
    exact_match = (test_side == eval_side) & (test_peak_locations == eval_peak_locations_on_true_side)
    exact_accuracy = np.mean(exact_match)
    logger.info(f"Exact Match Accuracy (side + peak): {np.sum(exact_match)}/{len(exact_match)} or {exact_accuracy:.4f} correct.")

    # Difference between predicted and true peak locations
    sample_max_force = np.max(np.max(np.abs(sample_forces), axis=2), axis=1)
    eval_max_force = np.max(np.max(np.abs(predicted_forces), axis=2), axis=1)
    location_differences = np.abs(sample_max_force - eval_max_force)
    mean_difference = np.mean(location_differences)
    logger.info(f"Mean Absolute Difference in Peak Locations: {mean_difference:.4f} Newton.")

def run_inverse_model_evaluation(
    model_path: Path,
    data_path: Path,
    unloaded_model: InverseModel,
) -> None:
    """Load data, preprocess it, load the surrogate model, and evaluate its performance."""
    model: InverseModel | None = load_inverse_model(model_path, unloaded_model)
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

    _, _, x_test, _, _, y_test = splitting(
        force_profiles,
        final_output_densities,
        random_state=1,
    )
    if x_test is None or y_test is None:
        logger.error("Failed to split the data into validation sets.")
        return

    y_test_tensor = torch.tensor(
    y_test, dtype=torch.float32, device=next(model.parameters()).device)

    with torch.no_grad():
        validation_predictions = model(y_test_tensor)

    # Convert predicted parameters → 3x10 force profiles
    validation_predictions_np = validation_predictions.cpu().numpy()

    # convert [zeroes area with one at peak to peak and side location]
    side_location = np.zeros(validation_predictions_np.shape[0])
    peak_location = np.zeros(validation_predictions_np.shape[0])
    for i in range(validation_predictions_np.shape[0]):
        peak_index = np.argmax(validation_predictions_np[i])
        peak_location[i] = peak_index % 10
        side_location[i] = peak_index // 10

    predictions = np.zeros((validation_predictions_np.shape[0], 3))
    predictions[:, 0] = peak_location[:]
    predictions[:, 1] = side_location[:]
    predictions[:, 2] = validation_predictions_np[:, 30]
    reconstructed_forces = np.array([
        params_to_force_profile(
            int(np.clip(pred[0], 0, 9)),
            int(np.clip(np.round(pred[1]), 0, 2)),
            float(pred[2]),
        )
        for pred in predictions
    ])

    inverse_model_metrics(sample_forces=x_test,
                          predicted_forces=reconstructed_forces)

    # Calculate similarity metrics
    surrogate_forwarder = SurrogateForwarder(
        surrogate_model_path=Path("/home/gijs/Desktop/Thesis/data/models/trained_model_new_data_1.pth"),
        density_shape=(10, 10),
        model_class=ReversedSurrogateModel,
    )
    if surrogate_forwarder.surrogate_model is None:
        logger.error("Failed to load the surrogate model for forward pass.")
        return
    ssim = np.zeros(reconstructed_forces.shape[0])
    mse = np.zeros(reconstructed_forces.shape[0])
    reconstructed_density = np.zeros((reconstructed_forces.shape[0], 10, 10))
    for sample in range(reconstructed_forces.shape[0]):
        reconstructed_density[sample] = surrogate_forwarder.forward_pass(reconstructed_forces[sample])
        ssim[sample] = calculate_similarity(reference_matrix=y_test[sample],
                         comparison_matrix=reconstructed_density[sample],
                         method="ssim")
        mse[sample] = calculate_similarity(reference_matrix=y_test[sample],
                        comparison_matrix=reconstructed_density[sample],
                        method="mse")
    logger.info(f"Average SSIM over validation set: {np.mean(ssim):.6f}")
    logger.info(f"Average MSE over validation set: {np.mean(mse):.6f}")

    for _ in range(100):
        k = np.random.randint(0, len(x_test))
        logger.info(f"Sample {k}: SSIM = {ssim[k]:.6f}, MSE = {mse[k]:.6f}")
        logger.info(f"Original Force Profile: {x_test[k]}")
        logger.info(f"Reconstructed Force Profile: {reconstructed_forces[k]}")
        plot_inverse_model(
            reconstructed_density[k],
            y_test[k],
            reconstructed_forces[k],
            x_test[k],
        )
    return

def load_inverse_model(
    model_path: Path,
    model: InverseModel,
) -> InverseModel | None:
    """Load the inverse surrogate model and its normalization parameters from a file.

    Args:
    ----
        model_path (Path): The path to the model file.
        model (InverseModel): The model instance to be loaded.

    Returns:
    -------
        tuple: A tuple containing the model, input mean, input std, output mean, and output std.

    """
    try:
        model.load_model(model_path)
        model.eval()
        return model
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None


def plot_inverse_model(
    predicted_matrix: np.ndarray,
    actual_matrix: np.ndarray,
    predicted_force_profile: np.ndarray,
    original_force_profile: np.ndarray,
) -> plt.Figure:
    """Compare the surrogate model's predictions with the actual validation data.

    Args:
    ----
        predicted_matrix (np.ndarray): Predicted density matrix from the surrogate model.
        actual_matrix (np.ndarray): Actual density matrix from the validation set.
        predicted_force_profile (np.ndarray): Force profile predicted by the inverse model.
        original_force_profile (np.ndarray): Original force profile from the validation set.

    """
    # Plot original, predicted, and difference matrices side by side (1 row, 3 columns)
    min_true_value: float = 0.01
    max_true_value: float = 1.74

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_density_matrix(
        actual_matrix,
        original_force_profile,
        "Original Density Matrix",
        axes[0],
        (min_true_value, max_true_value),
    )
    plot_density_matrix(
        predicted_matrix,
        predicted_force_profile,
        "Predicted Density Matrix",
        axes[1],
        (min_true_value, max_true_value),
    )
    plot_difference_matrix(
        predicted_matrix,
        actual_matrix,
        "Difference Matrix (Predicted - Actual) as Percentage",
        axes[2],
    )

    plt.tight_layout()
    plt.show()
    return figure

if __name__ == "__main__":
    model_path = Path("/home/gijs/Desktop/Thesis/data/inverse_model/trained_model.pth")
    data_path = Path("/home/gijs/Desktop/Thesis/data/raw/triangular/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InverseModel().to(device)
    run_inverse_model_evaluation(
        model_path=model_path,
        data_path=data_path,
        unloaded_model=model,
    )
