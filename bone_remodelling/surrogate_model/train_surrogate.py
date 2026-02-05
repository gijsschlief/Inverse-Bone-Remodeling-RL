"""Trainer script for the SurrogateModel. Note that SSIMS are calculated on normalized data here and thus lower than the true values."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from bone_remodelling.parameters import ConfigurationParameters
from bone_remodelling.surrogate_model.loader import SurrogatePredictor
from bone_remodelling.surrogate_model.neural_network import SurrogateModel
from bone_remodelling.surrogate_model.sanitizer import sanitize_data
from bone_remodelling.surrogate_model.splitter import load_and_split_data
from bone_remodelling.surrogate_model.train_parameters import (
    SurrogateTrainParameters,
)
from bone_remodelling.surrogate_model.trainer import SurrogateModelTrainer

logger = logging.getLogger(__name__)

def rescramble_for_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rescramble training and validation data for ensemble training.

    Args:
    ----
        x_train (np.ndarray): Training input features.
        y_train (np.ndarray): Training target labels.
        x_val (np.ndarray): Validation input features.
        y_val (np.ndarray): Validation target labels.
        random_state (int): Random seed for reproducibility.

    Returns:
    -------
        tuple[np.ndarray, np.ndarray]: Rescrambled training input features and target labels.

    """
    x_train_and_val = np.concatenate((x_train, x_val), axis=0)
    y_train_and_val = np.concatenate((y_train, y_val), axis=0)
    perm = np.random.RandomState(random_state).permutation(x_train_and_val.shape[0])
    x_train_and_val = x_train_and_val[perm]
    y_train_and_val = y_train_and_val[perm]
    split_index = x_train.shape[0]
    x_train_rescrambled = x_train_and_val[:split_index]
    y_train_rescrambled = y_train_and_val[:split_index]
    x_val_rescrambled = x_train_and_val[split_index:]
    y_val_rescrambled = y_train_and_val[split_index:]
    return x_train_rescrambled, y_train_rescrambled, x_val_rescrambled, y_val_rescrambled

def run_surrogate_training(
    data_file_path: Path,
    model_path: Path,
    random_state: int,
    ensemble_seed: int,
) -> None:
    """Train and evaluate the surrogate model."""
    logger.info(f"Loading data from {data_file_path}")
    (
        x_train_np,
        x_val_np,
        x_test_np,
        y_train_np,
        y_val_np,
        y_test_np,
    ) = load_and_split_data(data_file_path, random_state=random_state)

    x_train_np, y_train_np, x_val_np, y_val_np = rescramble_for_ensemble(
        x_train_np,
        y_train_np,
        x_val_np,
        y_val_np,
        random_state=ensemble_seed,
    )

    logger.info("Sanitizing data...")
    x_train_np, y_train_np = sanitize_data(x_train_np, y_train_np)
    x_val_np, y_val_np = sanitize_data(x_val_np, y_val_np)
    x_test_np, y_test_np = sanitize_data(x_test_np, y_test_np)

    logger.info(f"Training with random_state: {random_state} and ensemble_seed: {ensemble_seed}")
    train_parameters = SurrogateTrainParameters(
        model_path=model_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    predictor = SurrogatePredictor(SurrogateModel, train_parameters)
    surrogate_trainer = SurrogateModelTrainer(predictor, train_parameters)
    surrogate_trainer.time_training((x_train_np, y_train_np), (x_val_np, y_val_np))

def cli(
    configuration_parameters: ConfigurationParameters,
    cli_args: list[str],
) -> None:
    """CLI entry point for training the surrogate model.

    Args:
    ----
        configuration_parameters (ConfigurationParameters): Configuration parameters including output directory.
        cli_args (list[str]): Additional CLI arguments (not used here).

    """
    data_file_path = configuration_parameters.output_dir / Path("raw")
    model_path = configuration_parameters.output_dir / Path(f"surrogate_models/model_{configuration_parameters.seed}.pth")
    model_path = model_path.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train the surrogate model.")
    parser.add_argument(
        "--ensemble-seed",
        choices=range(0, 10000),
        type=int,
        default=configuration_parameters.seed,
        help="Ensemble seed for training.",
    )
    args = parser.parse_args(cli_args)
    run_surrogate_training(data_file_path, model_path, random_state=configuration_parameters.seed, ensemble_seed=args.ensemble_seed)

if __name__ == "__main__":
    # Developer convenience entry point.
    # For reproducible runs, use the unified CLI (main.py).
    config = ConfigurationParameters(output_dir=Path(__file__).parent.parent.parent / Path("data"))
    model_path = config.output_dir / Path(f"surrogate_models/model_{config.seed}.pth")
    data_file_path = config.output_dir / Path("raw")
    run_surrogate_training(data_file_path, model_path, random_state=config.seed, ensemble_seed=config.seed)
    logger.info("All training runs completed.")
    logger.info("Final model saved at: %s", model_path)
