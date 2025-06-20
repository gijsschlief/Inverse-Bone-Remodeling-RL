"""Evaluate the surrogate model's performance on validation data."""

import logging

from bone_remodeling.forward_model.data_reader import forward_data_reader

from surrogate_model.evaluator import average_similarity_score, validate_surrogate_model
from surrogate_model.loader import load_surrogate_model
from surrogate_model.neural_networks.advanced_neural_network import (
    AdvancedNNSurrogateModel,
)
from surrogate_model.splitter import splitting
from surrogate_model.visualizer import plot_surrogate_model


def main():
    """Load data, preprocess it, load the surrogate model, and evaluate its performance."""
    model = load_surrogate_model(
        "/home/gijs/Desktop/Thesis/data/models/trained_model_6.pth",
        AdvancedNNSurrogateModel,
    )
    if model is None:
        logging.error("Failed to load the surrogate model.")
        return
    _, force_profiles, final_output_densities = forward_data_reader(
        "/home/gijs/Desktop/Thesis/data/raw/"
    )
    if force_profiles is None or final_output_densities is None:
        logging.error("Failed to load the data.")
        return
    _, X_val, _, _, y_val, _ = splitting(force_profiles, final_output_densities)
    if X_val is None or y_val is None:
        logging.error("Failed to split the data into validation sets.")
        return

    predicted_matrices, true_matrices = validate_surrogate_model(model, X_val, y_val)
    if predicted_matrices is None or true_matrices is None:
        logging.error("Failed to validate the surrogate model.")
        return

    # FILTER THE MATRICES BEFORE MOVING ON!

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
