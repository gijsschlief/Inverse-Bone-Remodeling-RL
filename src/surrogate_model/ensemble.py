"""Loads an ensemble of surrogate models and returns mean and standard deviation predictions."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bone_remodeling.src.forward_model.density_visualizer import plot_density_matrix
from bone_remodeling.src.surrogate_model.loader import SurrogateModelLoader
from bone_remodeling.src.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)
from bone_remodeling.src.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.src.surrogate_model.normalizor import (
    load_normalization_params,
    normalize_data,
    unnormalize_data,
)
from bone_remodeling.src.surrogate_model.trainer import load_and_split_data
from bone_remodeling.src.surrogate_model.visualizer import plot_difference_matrix

logger = logging.getLogger(__name__)


def load_ensemble_models(
    model_paths: list[Path],
    model_class: type[SurrogateModel],
    model_loader: type[SurrogateModelLoader],
) -> tuple[
    list[type[SurrogateModel]],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """Load multiple surrogate models from specified paths."""
    models = []
    x_means = []
    x_stds = []
    y_means = []
    y_stds = []

    for model_path in model_paths:
        logger.info(f"Loading model from {model_path}")
        model = model_loader(model_path, model_class).model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        x_mean, x_std, y_mean, y_std = load_normalization_params(
            path=model_path.with_suffix(".npz"),
        )

        models.append(model)
        x_means.append(x_mean)
        x_stds.append(x_std)
        y_means.append(y_mean)
        y_stds.append(y_std)

    return models, x_means, x_stds, y_means, y_stds


def predict_with_ensemble(
    models: list[SurrogateModel],
    x: np.ndarray,
    x_normalizations: tuple[list[np.ndarray], list[np.ndarray]],
    y_normalizations: tuple[list[np.ndarray], list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict using an ensemble of models and return mean and standard deviation."""
    predictions = []
    x_means, x_stds = x_normalizations
    y_means, y_stds = y_normalizations
    for model, x_mean, x_std, y_mean, y_std in zip(
        models, x_means, x_stds, y_means, y_stds,
    ):
        prediction = surrogate_model_forward(model, (x_mean, x_std), (y_mean, y_std), x)
        predictions.append(prediction)

    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    std_prediction = np.std(predictions, axis=0)

    return mean_prediction, std_prediction


def surrogate_model_forward(
    model: type[SurrogateModel],
    x_normalization: tuple[np.ndarray, np.ndarray],
    y_normalization: tuple[np.ndarray, np.ndarray],
    force_profiles: np.ndarray,
) -> np.ndarray:
    """Forward pass through the surrogate model.

    Args:
    ----
        model (SurrogateModel): The surrogate model to use for prediction.
        x_normalization (tuple[np.ndarray, np.ndarray]): Mean and std for input normalization.
        y_normalization (tuple[np.ndarray, np.ndarray]): Mean and std for output unnormalization.
        force_profiles (np.ndarray): Input force profiles.

    Returns:
    -------
        np.ndarray: The predicted density from the surrogate model.

    """
    x_mean, x_std = x_normalization
    force_profile, _, _ = normalize_data(force_profiles, x_mean, x_std)

    # forward pass through the surrogate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        # Support batch predictions
        force_profile_tensor = torch.from_numpy(
            force_profile.reshape(force_profile.shape[0], -1).astype(np.float32)
            if force_profile.ndim > 1
            else force_profile.reshape(1, -1).astype(np.float32),
        ).to(device)

        density_tensor = model(force_profile_tensor)

        y_mean, y_std = y_normalization
        density_np = density_tensor.detach().cpu().numpy()
        surrogate_density: np.ndarray = unnormalize_data(density_np, y_mean, y_std)
    return surrogate_density


def plot_ensemble(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    true_matrices: np.ndarray,
    force_profiles: np.ndarray,
) -> None:
    """Plot the ensemble predictions against the true values.

    Args:
    ----
        mean_pred (np.ndarray): The mean predictions from the ensemble.
        std_pred (np.ndarray): The standard deviation of the predictions from the ensemble.
        true_matrices (np.ndarray): The true values to compare against.
        force_profiles (np.ndarray): The force profiles used for prediction.

    """
    axes = plt.subplots(2, 2, figsize=(12, 12))[1]
    plot_density_matrix(
        matrix=mean_pred,
        force_profile=force_profiles,
        title="Ensemble Mean Prediction",
        axis=axes[0, 0],
    )
    plot_density_matrix(
        matrix=true_matrices,
        force_profile=force_profiles,
        title="True Density",
        axis=axes[0, 1],
    )
    plot_difference_matrix(
        predicted_matrix=mean_pred,
        actual_matrix=true_matrices,
        title="Difference",
        axis=axes[1, 0],
    )
    plot_density_matrix(
        matrix=std_pred,
        force_profile=force_profiles,
        title="Prediction Uncertainty (Std Dev)",
        axis=axes[1, 1],
        color_scale=(0, 0.5),
    )
    plt.tight_layout()
    plt.show()


def main(
    model_paths: list[Path],
    model_class: type[SurrogateModel],
    model_loader: type[SurrogateModelLoader],
    data_file_path: Path,
    random_state: int = 0,
) -> None:
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
    models, x_means, x_stds, y_means, y_stds = load_ensemble_models(
        model_paths, model_class, model_loader,
    )

    # Make predictions on test set
    x_normalizations = (x_means, x_stds)
    y_normalizations = (y_means, y_stds)

    for _ in range(5):
        k = np.random.randint(0, len(x_test_np))
        logger.info(f"Predicting for test sample {k}")
        mean_pred, std_pred = predict_with_ensemble(
            models, x_test_np[k], x_normalizations, y_normalizations,
        )
        plot_ensemble(
            mean_pred.squeeze(), std_pred.squeeze(), y_test_np[k], x_test_np[k],
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
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

    main(
        model_paths,
        model_class=ReversedSurrogateModel,
        model_loader=SurrogateModelLoader,
        data_file_path=data_file_path,
        random_state=0,
    )
