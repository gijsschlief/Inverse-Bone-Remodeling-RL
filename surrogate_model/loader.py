import logging
from typing import Optional, Type
from pathlib import Path

import torch

from surrogate_model.neural_networks.advanced_neural_network import (
    AdvancedNNSurrogateModel,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SurrogateModelLoader:
    def __init__(
        self,
        model_path: str,
        model_class: Type[torch.nn.Module],
        auto_load: bool = True,
    ) -> None:
        """
        Initialize the loader with the path to the model and the model class.

        Args:
            model_path (str): Path to the .pth file containing the model weights.
            model_class (torch.nn.Module): The class of the model to be loaded.
        """
        self.model_path = model_path
        self.model_class = model_class
        self.model: Optional[torch.nn.Module] = None

        if auto_load:
            self.load()

    def load(self) -> None:
        """
        Load the surrogate model from the .pth file.
        """
        self.model = self.model_class()
        self.model.load_state_dict(
            torch.load(self.model_path, map_location="cpu", weights_only=False)
        )
        self.model.eval()  # Set the model to evaluation mode

    def forward(self, data_points: torch.Tensor) -> torch.Tensor:
        """
        Run forward estimation on the given data points.

        Args:
            data_points (torch.Tensor): Input data points for the model.

        Returns:
            torch.Tensor: Model predictions.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")

        with torch.no_grad():  # Disable gradient computation for inference
            predictions = self.model(data_points)
        return predictions

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Call the forward method of the model.
        """
        return self.forward(x)


def load_surrogate_model(
    model_path: str = "data/models/trained_model.pth",
    model_class: Type[torch.nn.Module] = AdvancedNNSurrogateModel,
) -> torch.nn.Module:
    """
    Main function to load data, preprocess it, load the surrogate model, and evaluate its performance.
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
    return model_loader.model


if __name__ == "__main__":
    model = load_surrogate_model()
    logging.info(f"Loaded model: {model}")
