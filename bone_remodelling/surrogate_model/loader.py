"""Module for loading and using a surrogate model for bone remodeling simulations."""

import logging
from pathlib import Path

import numpy as np
import torch

from bone_remodelling.surrogate_model.surrogate_parameters import (
    TrainParameters,
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
        model_class: type[torch.nn.Module],
        train_parameters: TrainParameters,
    ) -> None:
        """Initialize the loader with the path to the model and the model class.

        Args:
        ----
            model_class (Type[torch.nn.Module]): The class of the model to be loaded.
            train_parameters (SurrogateTrainParameters): Training parameters including model path and device.

        """
        self.model: torch.nn.Module = model_class()
        self.train_parameters = train_parameters
        self.model.to(self.train_parameters.device)

        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None

    def __call__(self, x_raw: np.ndarray) -> np.ndarray:
        """Call the object like a function: predictor(data)."""
        return self.predict(x_raw)

    def __repr__(self) -> str:
        """Show content of the SurrogatePredictor."""
        return f"SurrogatePredictor(model={self.train_parameters.model_path.name}, normalized={self.x_mean is not None})"

    def load_model(self) -> None:
        """Load the model from the specified path."""
        self.load_state_dictionary()
        self.load_normalization_params()

    def load_state_dictionary(self) -> None:
        """Load the state dictionary and set model to evaluation model."""
        self.model.load_state_dict(
            torch.load(
                self.train_parameters.model_path,
                map_location=self.train_parameters.device,
                weights_only=False,
            )
        )
        self.model.eval()

    def load_normalization_params(self) -> None:
        """Load normalization parameters associated with the model.

        Args:
        ----
            model_path (Path): Path to the .pth file containing the model weights.

        """
        normalization_path = self.train_parameters.model_path.with_suffix(".npz")

        if not normalization_path.exists():
            logger.warning(
                f"Normalization file not found at {normalization_path}. Using defaults."
            )
            return

        with np.load(normalization_path) as data:
            self.x_mean = data.get("x_mean").copy() if "x_mean" in data else None
            self.x_std = data.get("x_std").copy() if "x_std" in data else None
            self.y_mean = data.get("y_mean").copy() if "y_mean" in data else None
            self.y_std = data.get("y_std").copy() if "y_std" in data else None

    def load_history(self) -> dict[str, np.ndarray]:
        """Load training history associated with the model.

        Returns
        -------
            dict[str, np.ndarray]: A dictionary containing training history arrays.

        """
        history_path = self.train_parameters.model_path.with_suffix(".npz")

        if not history_path.exists():
            logger.warning(
                f"History file not found at {history_path}. Returning empty history."
            )
            return {}

        with np.load(history_path) as data:
            return {
                key: data[key].copy()
                for key in data.files
                if key not in {"x_mean", "x_std", "y_mean", "y_std"}
            }

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        """Run forward estimation on the given data points.

        Args:
        ----
            x_raw (np.ndarray): Input data points for the model.

        Returns:
        -------
            np.ndarray: Model predictions.

        """
        if self.x_mean is None or self.x_std is None:
            x_normalised = torch.tensor(x_raw, dtype=torch.float32).to(
                self.train_parameters.device
            )
        else:
            x_normalised_array = self.normalize_input(x_raw)
            x_normalised = torch.tensor(x_normalised_array, dtype=torch.float32).to(
                self.train_parameters.device
            )
        with torch.no_grad():
            y_normalised = self.model(x_normalised)
        if self.y_mean is None or self.y_std is None:
            return y_normalised.cpu().numpy()
        return self.unnormalize_output(y_normalised.cpu().numpy())

    def set_input_normalize(
        self,
        x_raw: np.ndarray,
    ) -> np.ndarray:
        """Set normalization parameters and normalize input data."""
        self.x_mean = np.mean(x_raw, axis=0)
        self.x_std = np.std(x_raw, axis=0) + 1e-8  # Prevent division by zero
        return (x_raw - self.x_mean) / self.x_std

    def set_output_normalize(
        self,
        y_raw: np.ndarray,
    ) -> np.ndarray:
        """Set normalization parameters and normalize output data."""
        self.y_mean = np.mean(y_raw, axis=0)
        self.y_std = np.std(y_raw, axis=0) + 1e-8  # Prevent division by zero
        return (y_raw - self.y_mean) / self.y_std

    def normalize_input(
        self,
        x_raw: np.ndarray,
    ) -> np.ndarray:
        """Normalize input data."""
        if self.x_mean is None or self.x_std is None:
            logger.warning(
                "Input normalization parameters are not set. Returning raw input."
            )
            return x_raw
        return (x_raw - self.x_mean) / self.x_std

    def normalize_output(
        self,
        y_raw: np.ndarray,
    ) -> np.ndarray:
        """Normalize output data."""
        if self.y_mean is None or self.y_std is None:
            logger.warning(
                "Output normalization parameters are not set. Returning raw output."
            )
            return y_raw
        return (y_raw - self.y_mean) / self.y_std

    def unnormalize_input(
        self,
        x_normalized: np.ndarray,
    ) -> np.ndarray:
        """Unnormalize input data."""
        if self.x_mean is None or self.x_std is None:
            logger.warning(
                "Input normalization parameters are not set. Returning normalized input."
            )
            return x_normalized
        return x_normalized * self.x_std + self.x_mean

    def unnormalize_output(
        self,
        y_normalized: np.ndarray,
    ) -> np.ndarray:
        """Unnormalize output data."""
        if self.y_mean is None or self.y_std is None:
            logger.warning(
                "Output normalization parameters are not set. Returning normalized output."
            )
            return y_normalized
        return y_normalized * self.y_std + self.y_mean

    def save_model_with_versioning(
        self, path: Path, history: dict[str, list[float]] | None = None
    ) -> None:
        """Save the model to a file, ensuring no overwriting of existing files."""
        try:
            if self.train_parameters.model_path.exists():
                base_path = self.train_parameters.model_path.with_suffix("")
                ext = self.train_parameters.model_path.suffix
                counter = 1
                while Path(f"{base_path}_{counter}{ext}").exists():
                    counter += 1
                path = Path(f"{base_path}_{counter}{ext}")
        except OSError as e:
            logger.warning(
                f"Safely saving model failed ({e}), overwriting existing file."
            )
        self.store_model_parameters(path, history=history)

    def store_model_parameters(
        self, path: Path, history: dict[str, list[float]] | None = None
    ) -> None:
        """Store the model parameters to the specified path."""
        torch.save(self.model.state_dict(), path)
        save_dict: dict[str, np.ndarray] = {}
        if self.x_mean is not None:
            save_dict["x_mean"] = self.x_mean
        if self.x_std is not None:
            save_dict["x_std"] = self.x_std
        if self.y_mean is not None:
            save_dict["y_mean"] = self.y_mean
        if self.y_std is not None:
            save_dict["y_std"] = self.y_std
        if history is not None:
            for key, values in history.items():
                save_dict[key] = np.array(values)
        np.savez_compressed(file=path.with_suffix(".npz"), **save_dict)  # type: ignore


def load_surrogate_models(
    model_class: type[torch.nn.Module],
    train_parameters: TrainParameters,
) -> list[SurrogatePredictor]:
    """Load any number of surrogate models from a specified folder."""
    model_folder = train_parameters.model_path.parent.resolve()
    logger.info(f"Loading all models from {model_folder}")

    predictors: list[SurrogatePredictor] = []

    for model_path in model_folder.glob("*.pth"):
        if not model_path.is_file() or model_path.stat().st_size == 0:
            logger.warning(f"Skipping invalid file: {model_path}")
            continue
        train_parameters.model_path = model_path
        try:
            predictor = SurrogatePredictor(
                model_class=model_class,
                train_parameters=train_parameters,
            )
            predictor.load_model()
            predictors.append(predictor)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")

    return predictors


def predict_with_surrogates(
    predictors: list[SurrogatePredictor],
    x: np.ndarray,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict using surrogate models and returns a list of the mean scores and the standard deviations."""
    if not predictors:
        raise ValueError("The list of predictors is empty. Cannot perform inference.")

    all_predictor_outputs = []
    for predictor in predictors:
        model_batches = []
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size]
            prediction = predictor(batch)
            model_batches.append(prediction)

        full_result_for_this_model = np.concatenate(model_batches, axis=0)
        all_predictor_outputs.append(full_result_for_this_model)

    mean_prediction = np.mean(all_predictor_outputs, axis=0)
    std_prediction = np.std(all_predictor_outputs, axis=0)
    return mean_prediction, std_prediction
