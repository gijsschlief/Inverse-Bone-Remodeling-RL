"""Trainer script for the InverseModel."""

import argparse
import logging
from pathlib import Path

import torch

from bone_remodelling.forward_data.forward_data_manager import ForwardDataManager
from bone_remodelling.inverse_model.inverse_neural_network import InverseModel
from bone_remodelling.inverse_model.inverse_parameters import (
    InverseTrainParameters,
)
from bone_remodelling.parameters import ConfigurationParameters
from bone_remodelling.surrogate_model.loader import SurrogatePredictor
from bone_remodelling.surrogate_model.sanitizer import sanitize_data
from bone_remodelling.surrogate_model.splitter import load_and_split_data
from bone_remodelling.surrogate_model.train_surrogate import rescramble_for_ensemble
from bone_remodelling.surrogate_model.trainer import SurrogateModelTrainer

logger = logging.getLogger(__name__)


def run_inverse_training(
    data_file_path: Path,
    model_path: Path,
    random_state: int,
    ensemble_seed: int,
) -> None:
    """Train and evaluate the inverse model."""
    forward_data_manager = ForwardDataManager(data_file_path)

    logger.info(f"Loading data from {data_file_path}")
    (
        y_train_np,
        y_val_np,
        y_test_np,
        x_train_np,
        x_val_np,
        x_test_np,
    ) = load_and_split_data(forward_data_manager, random_state=random_state)

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
    train_parameters = InverseTrainParameters(
        model_path=model_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    predictor = SurrogatePredictor(InverseModel, train_parameters)
    inverse_trainer = SurrogateModelTrainer(predictor, train_parameters, loss_function=torch.nn.MSELoss())
    inverse_trainer.time_training((x_train_np, y_train_np), (x_val_np, y_val_np))

def cli(
    configuration_parameters: ConfigurationParameters,
    cli_args: list[str],
) -> None:
    """CLI entry point for training the inverse model.

    Args:
    ----
        configuration_parameters (ConfigurationParameters): Configuration parameters including output directory.
        cli_args (list[str]): Additional CLI arguments (not used here).

    """
    data_file_path = configuration_parameters.output_dir / Path("raw", "triangular")
    model_path = configuration_parameters.output_dir / Path(f"inverse_models/model_{configuration_parameters.seed}.pth")
    model_path = model_path.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train the inverse model.")
    parser.add_argument(
        "--ensemble-seed",
        choices=range(0, 10000),
        type=int,
        default=configuration_parameters.seed,
        help="Ensemble seed for training.",
    )
    args = parser.parse_args(cli_args)
    run_inverse_training(data_file_path, model_path, random_state=configuration_parameters.seed, ensemble_seed=args.ensemble_seed)

if __name__ == "__main__":
    # Developer convenience entry point.
    # For reproducible runs, use the unified CLI (main.py).
    config = ConfigurationParameters(output_dir=Path(__file__).parent.parent.parent / Path("data"))
    model_path = config.output_dir / Path(f"inverse_models/model_{config.seed}.pth")
    data_file_path = config.output_dir / Path("raw", "triangular")
    run_inverse_training(data_file_path, model_path, random_state=config.seed, ensemble_seed=config.seed)
    logger.info("All training runs completed.")
    logger.info("Final model saved at: %s", model_path)
