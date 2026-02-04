"""Module for loading and using a surrogate model for bone remodeling simulations."""

import logging
from pathlib import Path

import numpy as np
import torch

from bone_remodelling.surrogate_model.neural_network import (
    SurrogateModel,
)

logger = logging.getLogger(__name__)


class SurrogatePredictor:
    """Class to load a surrogate model from a .pth file.

    This class handles the loading of a PyTorch model from a specified path and provides
    a method to run forward estimation on input data points.

    Attributes
    ----------
        model_path (str): Path to the .pth file containing the model weights.
        model_class (Type[torch.nn.Module]): The class of the model to be loaded.

    Methods
    -------
        predict(x_raw: np.ndarray) -> np.ndarray:
            Run forward estimation on the given data points.

    """

    def __init__(
        self,
        model_path: Path,
        model_class: type[torch.nn.Module],
    ) -> None:
        """Initialize the loader with the path to the model and the model class.

        Args:
        ----
            model_path (str): Path to the .pth file containing the model weights.
            model_class (torch.nn.Module): The class of the model to be loaded.

        """
        self.model_path = model_path
        self.model: torch.nn.Module = model_class()

        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None

        self.load_state_dictionary()
        self.load_normalization_params()

    def __call__(self, x_raw: np.ndarray) -> np.ndarray:
        """Call the object like a function: predictor(data)."""
        return self.predict(x_raw)

    def __repr__(self) -> str:
        """Show content of the SurrogatePredictor."""
        return f"SurrogatePredictor(model={self.model_path.name}, normalized={self.x_mean is not None})"

    def load_state_dictionary(self) -> None:
        """Load the state dictionary and set model to evaluation model."""
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu", weights_only=False))
        self.model.eval()

    def load_normalization_params(self) -> None:
        """Load normalization parameters associated with the model.

        Args:
        ----
            model_path (Path): Path to the .pth file containing the model weights.

        """
        normalization_path = self.model_path.with_suffix(".npz")
        if normalization_path.exists():
            data = np.load(normalization_path)
            self.x_mean = data.get("X_mean")
            self.x_std = data.get("X_std")
            self.y_mean = data.get("y_mean")
            self.y_std = data.get("y_std")

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        """Run forward estimation on the given data points.

        Args:
        ----
            x_raw (torch.Tensor): Input data points for the model.

        Returns:
        -------
            torch.Tensor: Model predictions.

        """
        if self.x_mean is None or self.x_std is None:
            x_normalised = torch.tensor(x_raw, dtype=torch.float32)
        else:
            x_normalised = self.normalize_input(
                x_raw,
                self.x_mean,
                self.x_std,
            )
            x_normalised = torch.tensor(x_normalised, dtype=torch.float32)
        with torch.no_grad():
            y_normalised = self.model.forward(x_normalised)
        if self.y_mean is None or self.y_std is None:
            return y_normalised.numpy()
        return self.unnormalize_output(
            y_normalised.numpy(),
            self.y_mean,
            self.y_std,
        )

    @staticmethod
    def normalize_input(
        x_raw: np.ndarray,
        x_mean: np.ndarray,
        x_std: np.ndarray,
    ) -> torch.Tensor:
        """Normalize input data.

        Args:
        ----
            x_raw (np.ndarray): Raw input data.
            x_mean (np.ndarray): Mean for normalization.
            x_std (np.ndarray): Standard deviation for normalization.

        Returns:
        -------
            torch.Tensor: Normalized input data.

        """
        x_normalized = (x_raw - x_mean) / x_std
        return torch.tensor(x_normalized, dtype=torch.float32)

    @staticmethod
    def unnormalize_output(
        y_normalized: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> np.ndarray:
        """Unnormalize output data.

        Args:
        ----
            y_normalized (np.ndarray): Normalized output data.
            y_mean (np.ndarray): Mean for unnormalization.
            y_std (np.ndarray): Standard deviation for unnormalization.

        Returns:
        -------
            np.ndarray: Unnormalized output data.

        """
        return y_normalized * y_std + y_mean


def load_surrogate_models(
    model_folder: Path,
    model_class: type[SurrogateModel],
) -> list[SurrogatePredictor]:
    """Load any number of surrogate models from a specified folder."""
    logger.info(f"Loading all models from {model_folder}")

    predictors: list[SurrogatePredictor] = []
    for model_path in model_folder.glob("*.pth"):
        if not model_path.is_file() or model_path.stat().st_size == 0:
            logger.warning(f"Skipping invalid file: {model_path}")
            continue

        try:
            predictor = SurrogatePredictor(
                model_path=model_path,
                model_class=model_class,
            )
            predictors.append(predictor)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")

    return predictors

def predict_with_surrogates(
    predictors: list[SurrogatePredictor],
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict using surrogate models and returns a list of the mean scores and the standard deviations."""
    if not predictors:
            raise ValueError("The list of predictors is empty. Cannot perform inference.")

    predictions = np.array([predictor(x) for predictor in predictors])
    mean_prediction = np.mean(predictions, axis=0)
    std_prediction = np.std(predictions, axis=0)
    return mean_prediction, std_prediction



if __name__ == "__main__":
    model_path = Path(__file__).parent.parent.parent / Path("data", "models", "surrogate.pth")
    predictor = SurrogatePredictor(model_path, SurrogateModel)
    logger.info(f"Loaded model: {predictor}")
