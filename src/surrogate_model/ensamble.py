"""Loads an ensemble of surrogate models and returns mean and standard deviation predictions."""

import logging
from pathlib import Path

import numpy as np
import torch

from bone_remodeling.src.surrogate_model.loader import SurrogateModelLoader
from bone_remodeling.src.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)
from bone_remodeling.src.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.src.surrogate_model.trainer import load_and_split_data
from bone_remodeling.src.surrogate_model.visualizer import plot_surrogate_model

logger = logging.getLogger(__name__)


def load_ensemble_models(model_paths: list[Path], model_class: type[SurrogateModel], model_loader: type[SurrogateModelLoader]) -> list[type[SurrogateModel]]:
    """Load multiple surrogate models from specified paths."""
    models = []
    for model_path in model_paths:
        logger.info(f"Loading model from {model_path}")
        model = model_loader(model_path, model_class).model
        models.append(model)
    return models


def predict_with_ensemble(
    models: list[SurrogateModel],
    x: np.ndarray,
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


def main(model_paths: list[Path], model_class: type[SurrogateModel], model_loader: type[SurrogateModelLoader], data_file_path: Path, random_state: int = 0) -> None:
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
    models = load_ensemble_models(model_paths, model_class, model_loader)

    # Make predictions on test set
    mean_pred, std_pred = predict_with_ensemble(models, x_test_np)

    # Plot predictions
    plot_surrogate_model(
        predicted_matrices=mean_pred,
        true_matrices=std_pred,
        force_profiles=x_test_np,
        sample_count=20,
        show_plot=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,                      # Show INFO and above
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_paths = [
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_1.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_2.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_3.pth"),
        # Add more model paths as needed
    ]
    data_file_path = Path("/home/gijs/Desktop/Thesis/data/raw/")

    main(model_paths, model_class=ReversedSurrogateModel, model_loader=SurrogateModelLoader, data_file_path=data_file_path, random_state=0)
