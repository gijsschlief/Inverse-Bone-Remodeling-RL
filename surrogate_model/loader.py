"""Module for loading and using a surrogate model for bone remodeling simulations."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Type

import numpy as np
import torch
from bone_remodeling.surrogate_model.neural_networks.medium_nn import (
    MediumSurrogateModel,
)
from bone_remodeling.surrogate_model.normalizor import load_normalization_params

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SurrogateModelLoader:
    """Class to load a surrogate model from a .pth file.

    This class handles the loading of a PyTorch model from a specified path and provides
    a method to run forward estimation on input data points.

    Attributes
    ----------
        model_path (str): Path to the .pth file containing the model weights.
        model_class (Type[torch.nn.Module]): The class of the model to be loaded.
        model (Optional[torch.nn.Module]): The loaded model instance, if auto_load is True.
        auto_load (bool): Whether to automatically load the model upon initialization.

    Methods
    -------
        load() -> None:
            Loads the model from the specified path.
        forward(data_points: torch.Tensor) -> torch.Tensor:
            Runs forward estimation on the given data points.
        __call__(x: torch.Tensor) -> torch.Tensor:
            Calls the forward method of the model.
        __init__(model_path: str, model_class: Type[torch.nn.Module], auto_load: bool = True) -> None:
            Initializes the loader with the path to the model and the model class.

    """

    def __init__(
        self,
        model_path: Path,
        model_class: Type[torch.nn.Module],
        auto_load: bool = True,
    ) -> None:
        """Initialize the loader with the path to the model and the model class.

        Args:
        ----
            model_path (str): Path to the .pth file containing the model weights.
            model_class (torch.nn.Module): The class of the model to be loaded.
            auto_load (bool): Whether to automatically load the model upon initialization.

        """
        self.model_path = model_path
        self.model_class = model_class
        self.model: Optional[torch.nn.Module] = None

        if auto_load:
            self.load()

    def load(self) -> None:
        """Load the surrogate model from the .pth file."""
        self.model = self.model_class()
        self.model.load_state_dict(
            torch.load(self.model_path, map_location="cpu", weights_only=False)
        )
        self.model.eval()  # Set the model to evaluation mode

    def forward(self, data_points: torch.Tensor) -> torch.Tensor:
        """Run forward estimation on the given data points.

        Args:
        ----
            data_points (torch.Tensor): Input data points for the model.

        Returns:
        -------
            torch.Tensor: Model predictions.

        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")

        with torch.no_grad():  # Disable gradient computation for inference
            predictions = self.model(data_points)
        return predictions

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Call the forward method of the model."""
        return self.forward(x)


def load_surrogate_model(
    model_path: str = "data/models/trained_model.pth",
    model_class: Type[torch.nn.Module] = MediumSurrogateModel,
) -> Tuple[
    torch.nn.Module | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Load data, preprocess it, load the surrogate model, and evaluate its performance.

    Args:
    ----
        model_path (str): Path to the .pth file containing the model weights.
        model_class (Type[torch.nn.Module]): The class of the model to be loaded.

    Returns:
    -------
        Tuple[torch.nn.Module | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
            The loaded model and normalization parameters (if available).

    """
    path = Path(model_path)

    # Check if the model path is valid
    if not path.is_absolute():
        code_dir = (
            Path(__file__).resolve().parent.parent.parent
        )  # Resolve Thesis_code directory dynamically
        path = code_dir / path
        if not path.is_absolute():
            logging.error(
                f"Failed to resolve absolute path for model file: {model_path}"
            )
            raise ValueError(
                f"Failed to resolve absolute path for model file: {model_path}"
            )
    if not path.is_file():
        logging.error(f"Model file does not exist: {model_path}")
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if not path.suffix == ".pth":
        logging.error(f"Invalid model file format: {model_path}. Expected a .pth file.")
        raise ValueError(
            f"Invalid model file format: {model_path}. Expected a .pth file."
        )
    if not path.is_absolute():
        logging.error(f"Model file path is not absolute: {model_path}")
        raise ValueError(f"Model file path is not absolute: {model_path}")
    logging.info(f"Loading model from {model_path} with class {model_class.__name__}")

    # Load the surrogate model
    model_loader = SurrogateModelLoader(model_path=path, model_class=model_class)

    # If the model has normalization parameters, load them
    if path.with_suffix(".npz").exists():
        X_mean, X_std, y_mean, y_std = load_normalization_params(
            path=path.with_suffix(".npz")
        )
        return model_loader.model, X_mean, X_std, y_mean, y_std

    return model_loader.model, None, None, None, None


if __name__ == "__main__":
    model = load_surrogate_model()
    logging.info(f"Loaded model: {model}")
