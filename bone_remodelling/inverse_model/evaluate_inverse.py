"""Evaluate the inverse model's performance on validation data."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from bone_remodelling.forward_data.forward_data_manager import (
    ForwardDataManager,
)
from bone_remodelling.inverse_model.inverse_neural_network import InverseModel
from bone_remodelling.inverse_model.inverse_parameters import InverseTrainParameters
from bone_remodelling.inverse_model.triangular_to_params_converter import (
    force_profile_to_params,
    params_to_force_profile,
    reshape_features_back_to_params,
    reshape_input_features_for_model,
)
from bone_remodelling.parameters import ConfigurationParameters
from bone_remodelling.rl_model.reward_calculation import calculate_similarity
from bone_remodelling.surrogate_model.loader import (
    SurrogatePredictor,
    load_surrogate_models,
    predict_with_surrogates,
)
from bone_remodelling.surrogate_model.neural_network import SurrogateModel
from bone_remodelling.surrogate_model.sanitizer import sanitize_data
from bone_remodelling.surrogate_model.splitter import splitting
from bone_remodelling.surrogate_model.surrogate_parameters import (
    SurrogateTrainParameters,
)
from bone_remodelling.surrogate_model.visualizer import plot_surrogate

logger = logging.getLogger(__name__)


def inverse_model_metrics(
    sample_forces: np.ndarray,
    predicted_forces: np.ndarray,
) -> None:
    """Calculate metrics for the inverse model predictions. Looks at the force side, exact location and magnitude differences."""
    # Force Side Selection
    test_side = np.argmax(np.max(np.abs(sample_forces), axis=2), axis=1)
    eval_side = np.argmax(np.max(np.abs(predicted_forces), axis=2), axis=1)
    side_accuracy = np.sum(test_side == eval_side) / len(test_side)
    logger.info(
        f"Force Side Selection Accuracy: {np.sum(test_side == eval_side)}/{len(test_side)} or {side_accuracy:.4f} correct.",
    )

    # Correct per-sample peak location extraction
    test_peak_locations = np.array(
        [
            np.argmax(np.abs(sample_forces[i, test_side[i], :]))
            for i in range(len(sample_forces))
        ],
    )

    eval_peak_locations_on_true_side = np.array(
        [
            np.argmax(np.abs(predicted_forces[i, test_side[i], :]))
            for i in range(len(predicted_forces))
        ],
    )
    exact_match = (test_side == eval_side) & (
        test_peak_locations == eval_peak_locations_on_true_side
    )
    exact_accuracy = np.mean(exact_match)
    logger.info(
        f"Exact Match Accuracy (side + peak): {np.sum(exact_match)}/{len(exact_match)} or {exact_accuracy:.4f} correct.",
    )

    # Difference between predicted and true peak locations
    sample_max_force = np.max(np.max(np.abs(sample_forces), axis=2), axis=1)
    eval_max_force = np.max(np.max(np.abs(predicted_forces), axis=2), axis=1)
    location_differences = np.abs(sample_max_force - eval_max_force)
    mean_difference = np.mean(location_differences)
    logger.info(
        f"Mean Absolute Difference in Peak Locations: {mean_difference:.4f} Newton.",
    )


def inverse_model_evaluation(
    data_path: Path,
    model_class: type[torch.nn.Module],
    train_parameters: InverseTrainParameters,
    random_state: int,
    metric: str = "ssim",
) -> None:
    """Load data, preprocess it, load the inverse model, and evaluate its performance."""
    np.random.seed(random_state)
    raw_path = data_path / Path("raw", "triangular")
    data = ForwardDataManager(raw_path).load_directory()
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

    y_train, y_val, y_test, x_train, x_val, x_test = splitting(
        force_profiles,
        final_output_densities,
        random_state=random_state,
    )

    # Load inverse models
    predictors = load_surrogate_models(model_class, train_parameters)
    y_train_predicted = inverse_prediction(
        predictors,
        train_parameters,
        x_train,
        y_train,
    )
    y_val_predicted = inverse_prediction(predictors, train_parameters, x_val, y_val)
    y_test_predicted = inverse_prediction(predictors, train_parameters, x_test, y_test)

    logger.info("Calculating inverse model metrics on the test set...")
    inverse_model_metrics(sample_forces=y_test, predicted_forces=y_test_predicted)

    try:
        evaluate_predictions_with_surrogate(
            surrogate_model_path=data_path / Path("surrogate_models/model.pth"),
            device=train_parameters.device,
            force_predictions=(y_train_predicted, y_val_predicted, y_test_predicted),
            true_densities=(x_train, x_val, x_test),
            metric=metric,
            true_forces=(y_train, y_val, y_test),
        )
    except OSError as e:
        logger.error(f"Could not load surrogate model for evaluation: {e}")
        return


def inverse_prediction(
    predictors: list[SurrogatePredictor],
    train_parameters: InverseTrainParameters,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Evaluate the predicted parameters by converting them back to force profiles and using the surrogate model to predict the resulting densities, then comparing those to the true densities."""
    y = np.array([force_profile_to_params(fp) for fp in y])
    y = reshape_input_features_for_model(y)
    y_predicted, _ = predict_with_surrogates(
        predictors,
        x,
        batch_size=train_parameters.batch_size,
    )
    y_predicted = reshape_features_back_to_params(y_predicted)
    return np.array(
        [
            params_to_force_profile(
                int(y_predicted[i][0]),
                int(y_predicted[i][1]),
                y_predicted[i][2],
            )
            for i in range(len(y_predicted))
        ],
    )


def evaluate_predictions_with_surrogate(
    surrogate_model_path: Path,
    device: torch.device,
    force_predictions: tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray,
    true_densities: tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray,
    metric: str,
    true_forces: tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray,
) -> None:
    """Evaluate the predicted parameters by converting them back to force profiles and using the surrogate model to predict the resulting densities, then comparing those to the true densities."""
    surrogate_train_parameters = SurrogateTrainParameters(
        model_path=surrogate_model_path,
        device=device,
    )

    surrogate_predictors = load_surrogate_models(
        SurrogateModel,
        surrogate_train_parameters,
    )

    if isinstance(true_forces, np.ndarray):
        true_forces = (true_forces, true_forces, true_forces)
    if isinstance(force_predictions, np.ndarray):
        force_predictions = (force_predictions, force_predictions, force_predictions)
    if isinstance(true_densities, np.ndarray):
        true_densities = (true_densities, true_densities, true_densities)
    _, _, x_test = true_forces
    x_train_pred, x_val_pred, x_test_pred = force_predictions

    y_train, y_val, y_test = true_densities

    y_train_predicted, _ = predict_with_surrogates(
        surrogate_predictors,
        x_train_pred,
        batch_size=surrogate_train_parameters.batch_size,
    )
    y_val_predicted, _ = predict_with_surrogates(
        surrogate_predictors,
        x_val_pred,
        batch_size=surrogate_train_parameters.batch_size,
    )
    y_test_predicted, y_test_std = predict_with_surrogates(
        surrogate_predictors,
        x_test_pred,
        batch_size=surrogate_train_parameters.batch_size,
    )

    train_ssim = [
        calculate_similarity(
            y_train_predicted[i],
            y_train[i],
            baseline=0.1,
            threshold=0.5,
            method=metric,
        )
        for i in range(len(y_train_predicted))
    ]
    logger.info(f"Train {metric.upper()}: {np.mean(train_ssim):.4f}")

    val_ssim = [
        calculate_similarity(
            y_val_predicted[i],
            y_val[i],
            baseline=0.1,
            threshold=0.5,
            method=metric,
        )
        for i in range(len(y_val_predicted))
    ]
    logger.info(f"Validation {metric.upper()}: {np.mean(val_ssim):.4f}")

    test_ssim = [
        calculate_similarity(
            y_test_predicted[i],
            y_test[i],
            baseline=0.1,
            threshold=0.5,
            method=metric,
        )
        for i in range(len(y_test_predicted))
    ]
    logger.info(f"Test {metric.upper()}: {np.mean(test_ssim):.4f}")

    # Visualize some results from the test set
    worst_index = np.argmin(test_ssim)
    plot_surrogate(
        y_test_predicted[worst_index].squeeze(),
        y_test_std[worst_index].squeeze(),
        y_test[worst_index],
        x_test[worst_index],
        x_test_pred[worst_index],
    )

    for k in np.random.choice(len(x_test), size=5, replace=False):
        plot_surrogate(
            y_test_predicted[k].squeeze(),
            y_test_std[k].squeeze(),
            y_test[k],
            x_test[k],
            x_test_pred[k],
        )


def cli(config: ConfigurationParameters, cli_args: list[str]) -> None:
    """Command-line interface for inverse model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the inverse model.")
    parser.add_argument(
        "--batch-size",
        choices=range(1, 10_000),
        default=1024,
        type=int,
        help="Batch size for inverse model evaluation.",
    )
    parser.add_argument(
        "--metric",
        choices=["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"],
        default="ssim",
        type=str,
        help="Metric for inverse model evaluation.",
    )
    args = parser.parse_args(cli_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = (config.output_dir / Path("inverse_models", "model.pth")).resolve()
    train_parameters = InverseTrainParameters(
        model_path=model_path,
        device=device,
        batch_size=args.batch_size,
    )
    inverse_model_evaluation(
        data_path=config.output_dir,
        model_class=InverseModel,
        train_parameters=train_parameters,
        random_state=config.seed,
        metric=args.metric,
    )


if __name__ == "__main__":
    # Developer convenience entry point.
    # For reproducible runs, use the unified CLI (main.py).
    config = ConfigurationParameters(
        output_dir=Path(__file__).parent.parent.parent / Path("data", "triangular"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = (config.output_dir / Path("inverse_models", "model.pth")).resolve()
    train_parameters = InverseTrainParameters(model_path=model_path, device=device)
    inverse_model_evaluation(
        data_path=config.output_dir,
        model_class=InverseModel,
        train_parameters=train_parameters,
        random_state=config.seed,
    )
