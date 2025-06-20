"""Evaluate the surrogate model's performance on validation data."""

import logging

from bone_remodeling.forward_model.data_reader import forward_data_reader

from surrogate_model.evaluator import evaluate_surrogate_model
from surrogate_model.loader import load_surrogate_model
from surrogate_model.neural_networks.advanced_neural_network import (
    AdvancedNNSurrogateModel,
)
from surrogate_model.splitter import splitting
from surrogate_model.visualizer import plot_surrogate_model


def main():
    """Load data, preprocess it, load the surrogate model, and evaluate its performance."""
    # Load the surrogate model and data
    model = load_surrogate_model(
        "/home/gijs/Desktop/Thesis/data/models/trained_model_6.pth",
        AdvancedNNSurrogateModel,
    )
    _, force_profiles, final_output_densities = forward_data_reader(
        "/home/gijs/Desktop/Thesis/data/raw/"
    )
    _, X_val, _, _, y_val, _ = splitting(force_profiles, final_output_densities)

    val_logits, y_val_tensor, average_similarity = evaluate_surrogate_model(
        model, X_val, y_val
    )

    logging.info(f"The average similarity = {average_similarity}")

    plot_surrogate_model(val_logits, y_val_tensor, sample_count=3, show_plot=True)
    return


if __name__ == "__main__":
    main()
