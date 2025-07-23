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
from bone_remodeling.src.surrogate_model.normalizor import load_normalization_params

logger = logging.getLogger(__name__)


def load_ensemble_models(model_paths: list[Path], model_class: type[SurrogateModel], model_loader: type[SurrogateModelLoader]) -> tuple[list[type[SurrogateModel]], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Load multiple surrogate models from specified paths."""
    models = []
    x_means = []
    x_stds = []
    y_means = []
    y_stds = []

    for model_path in model_paths:
        logger.info(f"Loading model from {model_path}")
        model = model_loader(model_path, model_class).model

        x_mean, x_std, y_mean, y_std = load_normalization_params(
            path=model_path.with_suffix(".npz"),
        )

        models.append(model)
        x_means.append(x_mean)
        x_stds.append(x_std)
        y_means.append(y_mean)
        y_stds.append(y_stds)

    return models, x_means, x_stds, y_means, y_stds


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

def _surrogate_model_forward() -> np.ndarray:
    """Forward pass through the surrogate model.

    Args:
    ----
        force_profile (np.ndarray): The force profile applied to the bone.
        return_shape (tuple): The shape to return the predicted density.

    Returns:
    -------
        np.ndarray: The predicted density from the surrogate model.

    """
    # normalize
    if self.x_mean is not None and self.x_std is not None:
        force_profile, _, _ = normalize_data(
            self.force_profile,
            self.x_mean,
            self.x_std,
        )

    # forward pass through the surrogate model
    with torch.no_grad():
        force_profile_tensor = torch.from_numpy(
            force_profile.reshape(1, -1).astype(np.float32),
        ).to(self.device)

    assert (
        force_profile_tensor is not None
    ), "Force profile tensor is None after reshaping."
    model = cast(Module, self.surrogate_model)
    density_tensor = model(force_profile_tensor)
    assert (
        density_tensor is not None
    ), "Density tensor is None after model forward pass."

    # unnormalize
    if self.y_mean is not None and self.y_std is not None:
        density_tensor = unnormalize_data(density_tensor, self.y_mean, self.y_std)

    surrogate_density: np.ndarray = density_tensor.detach().cpu().numpy()
    return surrogate_density

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
        true_matrices=y_test_np,
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
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_4.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_5.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_6.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_7.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_8.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_9.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_10.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_11.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_12.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_13.pth"),
    ]
    data_file_path = Path("/home/gijs/Desktop/Thesis/data/raw/")

    main(model_paths, model_class=ReversedSurrogateModel, model_loader=SurrogateModelLoader, data_file_path=data_file_path, random_state=0)
