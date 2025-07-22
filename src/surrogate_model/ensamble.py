"""Loads an ensemble of surrogate models and returns mean and standard deviation predictions."""

import logging
from pathlib import Path

import numpy as np
import torch

from bone_remodeling.src.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)
from bone_remodeling.src.surrogate_model.trainer import load_and_split_data

logging.basicConfig(level=logging.INFO)


def load_ensemble_models(model_paths: list[Path]) -> list[SurrogateModel]:
    """Load multiple surrogate models from specified paths."""
    models = []
    for model_path in model_paths:
        logging.info(f"Loading model from {model_path}")
        model = SurrogateModel.load(model_path)
        models.append(model)
    return models


def predict_with_ensemble(
    models: list[SurrogateModel], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Predict using an ensemble of models and return mean and standard deviation."""
    predictions = []
    for model in models:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            pred = model(x_tensor).cpu().numpy()
            predictions.append(pred)

    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    std_prediction = np.std(predictions, axis=0)

    return mean_prediction, std_prediction


def main(model_paths: list[Path], data_file_path: Path, random_state: int = 0) -> None:
    """Load models, make predictions, and save results."""
    # Load data
    (
        x_train_np,
        x_val_np,
        x_test_np,
        y_train_np,
        y_val_np,
        y_test_np,
    ) = load_and_split_data(data_file_path, random_state=random_state)

    # Load models
    models = load_ensemble_models(model_paths)

    # Make predictions on test set
    mean_pred, std_pred = predict_with_ensemble(models, x_test_np)

    # Save predictions
    np.savez("ensemble_predictions.npz", mean=mean_pred, std=std_pred)
    logging.info("Predictions saved to ensemble_predictions.npz")


if __name__ == "__main__":
    model_paths = [
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_1.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_2.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_3.pth"),
        # Add more model paths as needed
    ]
    data_file_path = Path("/home/gijs/Desktop/Thesis/data/raw/")

    main(model_paths, data_file_path, random_state=0)
