"""Evaluate the surrogate model's performance on validation data."""

import logging
from pathlib import Path

import numpy as np

from bone_remodelling.forward_data.reader import forward_data_reader
from bone_remodelling.parameters import ConfigurationParameters
from bone_remodelling.rl_model.reward_calculation import calculate_similarity
from bone_remodelling.surrogate_model.loader import (
    load_surrogate_models,
    predict_with_surrogates,
)
from bone_remodelling.surrogate_model.neural_network import (
    SurrogateModel,
)
from bone_remodelling.surrogate_model.sanitizer import sanitize_data
from bone_remodelling.surrogate_model.splitter import splitting
from bone_remodelling.surrogate_model.visualizer import plot_surrogate

logger = logging.getLogger(__name__)


def surrogates_evaluation(
    data_path: Path,
    model_class: type[SurrogateModel],
    random_state: int = 0,
) -> None:
    """Load data, preprocess it, load the surrogate model, and evaluate its performance."""
    np.random.seed(random_state)
    raw_path = data_path / Path("raw")
    data = forward_data_reader(raw_path)
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
        random_state=random_state,
    )

    # Load surrogate models
    surrogate_path = data_path / Path("models")
    predictors = load_surrogate_models(surrogate_path, model_class)

    # Run the predictors on the three sets
    y_train_predicted, _ = predict_with_surrogates(predictors, x_train)
    y_val_predicted, _ = predict_with_surrogates(predictors, x_val)
    y_test_predicted, y_test_std = predict_with_surrogates(predictors, x_test)

    # Calculate average similarity scores
    train_ssim = [calculate_similarity(y_train_predicted[i], y_train[i], baseline=0.1, threshold=0.5, method="ssim") for i in range(len(y_train_predicted))]
    logger.info(f"Train SSIM: {np.mean(train_ssim):.4f}")

    val_ssim = [calculate_similarity(y_val_predicted[i], y_val[i], baseline=0.1, threshold=0.5, method="ssim") for i in range(len(y_val_predicted))]
    logger.info(f"Validation SSIM: {np.mean(val_ssim):.4f}")

    test_ssim = [calculate_similarity(y_test_predicted[i], y_test[i], baseline=0.1, threshold=0.5, method="ssim") for i in range(len(y_test_predicted))]
    logger.info(f"Test SSIM: {np.mean(test_ssim):.4f}")

    # Visualize some results from the test set
    worst_index = np.argmin(test_ssim)
    plot_surrogate(
        y_test_predicted[worst_index].squeeze(),
        y_test_std[worst_index].squeeze(),
        y_test[worst_index],
        x_test[worst_index],
    )

    for k in np.random.choice(len(x_test), size=5, replace=False):
        plot_surrogate(
            y_test_predicted[k].squeeze(),
            y_test_std[k].squeeze(),
            y_test[k],
            x_test[k],
        )

def cli(config: ConfigurationParameters) -> None:
    """Command-line interface for surrogate model evaluation."""
    surrogates_evaluation(
        data_path=config.output_dir,
        model_class=SurrogateModel,
        random_state=config.seed,
    )

if __name__ == "__main__":
    # Developer convenience entry point.
    # For reproducible runs, use the unified CLI (main.py).
    config = ConfigurationParameters()
    surrogates_evaluation(data_path=config.output_dir, model_class=SurrogateModel, random_state=config.seed)
