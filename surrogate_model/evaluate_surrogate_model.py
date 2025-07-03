"""Evaluate the surrogate model's performance on validation data."""

import logging

from bone_remodeling.forward_model.data_reader import forward_data_reader
from bone_remodeling.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)

from surrogate_model.evaluator import average_similarity_score, validate_surrogate_model
from surrogate_model.loader import load_surrogate_model
from surrogate_model.normalizor import (
    normalize_data,
    unnormalize_data,
)
from surrogate_model.sanitizer import sanitize_data, sanitize_matrices
from surrogate_model.splitter import splitting
from surrogate_model.visualizer import plot_surrogate_model


def main() -> None:
    """Load data, preprocess it, load the surrogate model, and evaluate its performance."""
    model_and_normalization_params = load_surrogate_model(
        "/home/gijs/Desktop/Thesis/data/models/trained_model.pth",
        ReversedSurrogateModel,
    )
    if model_and_normalization_params is None:
        logging.error(
            "Failed to load the surrogate model and normalization parameters."
        )
        return
    model, x_mean, x_std, y_mean, y_std = model_and_normalization_params
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
    force_profiles, final_output_densities = sanitize_data(
        force_profiles, final_output_densities
    )

    x_train, x_val, x_test, y_train, y_val, y_test = splitting(
        force_profiles, final_output_densities, random_state=0
    )
    if x_val is None or y_val is None:
        logging.error("Failed to split the data into validation sets.")
        return

    # Normalize the validation data if normalization parameters are available
    if x_mean is not None and x_std is not None:
        x_val_unnormalized = x_val.copy()
        x_train, x_val, x_test, _, _ = normalize_data(
            x_train, x_val, x_test, x_mean, x_std
        )

    if y_mean is not None and y_std is not None:
        y_train, y_val, y_test, _, _ = normalize_data(
            y_train, y_val, y_test, y_mean, y_std
        )
        output_normalized = True
    else:
        output_normalized = False

    predicted_matrices, true_matrices = validate_surrogate_model(model, x_val, y_val)

    # predicted_matrices, true_matrices = sanitize_matrices(
    #    predicted_matrices, true_matrices
    # )

    if output_normalized:
        predicted_matrices = unnormalize_data(predicted_matrices, y_mean, y_std)
        true_matrices = unnormalize_data(true_matrices, y_mean, y_std)

    average_similarity = average_similarity_score(
        predicted_matrices, true_matrices, baseline=0.1, threshold=0.5, method="ssim"
    )

    logging.info(f"The average similarity = {average_similarity}")

    plot_surrogate_model(
        predicted_matrices=predicted_matrices,
        true_matrices=true_matrices,
        force_profiles=x_val_unnormalized,
        sample_count=3,
        show_plot=True,
    )
    return


if __name__ == "__main__":
    main()
