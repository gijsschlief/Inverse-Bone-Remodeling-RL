"""Trainer script for the SurrogateModel."""

import argparse
import logging
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import torch

from bone_remodelling.forward_data.forward_data_manager import ForwardDataManager
from bone_remodelling.parameters import ConfigurationParameters
from bone_remodelling.surrogate_model.loader import SurrogatePredictor
from bone_remodelling.surrogate_model.loss_function import combined_loss
from bone_remodelling.surrogate_model.neural_network import SurrogateModel
from bone_remodelling.surrogate_model.sanitizer import sanitize_data
from bone_remodelling.surrogate_model.splitter import load_and_split_data
from bone_remodelling.surrogate_model.surrogate_parameters import (
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
    return (
        x_train_rescrambled,
        y_train_rescrambled,
        x_val_rescrambled,
        y_val_rescrambled,
    )

def run_surrogate_training(
    data_file_path: Path,
    model_path: Path,
    random_state: int,
    ensemble_seed: int,
    *,
    tune: bool,
) -> None:
    """Train and evaluate the surrogate model."""
    forward_data_manager = ForwardDataManager(data_file_path)

    logger.info(f"Loading data from {data_file_path}")
    (
        x_train_np,
        x_val_np,
        x_test_np,
        y_train_np,
        y_val_np,
        y_test_np,
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

    logger.info(
        f"Training with random_state: {random_state} and ensemble_seed: {ensemble_seed}",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tune:
        logger.info("Running hyperparameter tuning...")
        best_parameters = hyperparameter_search(
            (x_train_np, y_train_np), (x_val_np, y_val_np), model_path, device,
        )
        train_parameters = SurrogateTrainParameters(
            model_path=model_path,
            device=device,
            learning_rate=best_parameters["learning_rate"],
            batch_size=best_parameters["batch_size"],
            weight_decay=best_parameters["weight_decay"],
            width=best_parameters["width"],
            depth=best_parameters["depth"],
            dropout=best_parameters["dropout"],
            patience_lr_scheduler=best_parameters["patience_lr_scheduler"],
            patience=best_parameters["patience_lr_scheduler"]*3,
        )
    else:
        train_parameters = SurrogateTrainParameters(
            model_path=model_path,
            device=device,
        )

    predictor = SurrogatePredictor(SurrogateModel, train_parameters)
    combined_loss_function = partial(combined_loss, loss_weight=train_parameters.alpha)
    surrogate_trainer = SurrogateModelTrainer(
        predictor,
        train_parameters,
        loss_function=combined_loss_function,
    )
    surrogate_trainer.time_training((x_train_np, y_train_np), (x_val_np, y_val_np))

def objective(trial: optuna.trial.Trial, train_data: tuple[np.ndarray, np.ndarray], validation_data: tuple[np.ndarray, np.ndarray], model_path: Path, device: torch.device) -> float:
    """Objective defined for the optuna training."""
    hyperparameters = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_int("batch_size", 16, 256),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "width": trial.suggest_categorical("width", [128, 256, 512, 1024, 2048]),
        "depth": trial.suggest_categorical("depth", [4, 6, 8, 10]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "patience_lr_scheduler": trial.suggest_int("patience_lr_scheduler", 5, 20),
    }

    trial_params = SurrogateTrainParameters(
        model_path=model_path,
        device=device,
        learning_rate=hyperparameters["learning_rate"],
        batch_size=hyperparameters["batch_size"],
        weight_decay=hyperparameters["weight_decay"],
        width=hyperparameters["width"],
        depth=hyperparameters["depth"],
        dropout=hyperparameters["dropout"],
        patience_lr_scheduler=hyperparameters["patience_lr_scheduler"],
        patience=hyperparameters["patience_lr_scheduler"]*3,
    )

    predictor = SurrogatePredictor(
        SurrogateModel,
        trial_params,
    )

    combined_loss_function = partial(combined_loss, loss_weight=trial_params.alpha)
    trainer = SurrogateModelTrainer(predictor, trial_params, combined_loss_function)

    try:
        trainer.train(train_data, validation_data, trial=trial)
    except optuna.exceptions.TrialPruned:
        # Re-raise so Optuna knows it was pruned
        raise

    return trainer.best_val_loss

def hyperparameter_search(
    train_data: tuple[np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray],
    model_path: Path,
    device: torch.device,
) -> dict:
    """Hyperparameter tuning function."""
    study_name = "surrogate_tuning"
    db_path = model_path.parent / Path(f"{study_name}.db")
    storage_name = f"sqlite:///{db_path.resolve()}"

    random_first_trials = 10
    warmup_epochs = 20

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=random_first_trials,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=random_first_trials,
            n_warmup_steps=warmup_epochs,
        ),
    )

    objective_function = partial(
        objective,
        train_data=train_data,
        validation_data=validation_data,
        model_path=model_path,
        device=device,
    )
    study.optimize(objective_function, n_trials=100)  # Start with 50-100 trials
    logger.info("Best Trial:")
    logger.info(study.best_trial.params)
    return study.best_params

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
    model_path = configuration_parameters.output_dir / Path(
        f"surrogate_models/model_{configuration_parameters.seed}.pth",
    )
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
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Hyperparameter tuning for the surrogate training.",
    )
    args = parser.parse_args(cli_args)
    run_surrogate_training(
        data_file_path,
        model_path,
        random_state=configuration_parameters.seed,
        ensemble_seed=args.ensemble_seed,
        tune=args.tune,
    )


if __name__ == "__main__":
    # Developer convenience entry point.
    # For reproducible runs, use the unified CLI (main.py).
    config = ConfigurationParameters(
        output_dir=Path(__file__).parent.parent.parent / Path("data"),
    )
    model_path = config.output_dir / Path(f"surrogate_models/model_{config.seed}.pth")
    data_file_path = config.output_dir / Path("raw")
    run_surrogate_training(
        data_file_path,
        model_path,
        random_state=config.seed,
        ensemble_seed=config.seed,
        tune=False,
    )
    logger.info("All training runs completed.")
    logger.info("Final model saved at: %s", model_path)
