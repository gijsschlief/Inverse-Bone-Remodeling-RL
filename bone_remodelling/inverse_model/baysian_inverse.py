"""Bayesian inverse model for bone remodeling."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bone_remodelling.forward_data.forward_data_manager import ForwardDataManager
from bone_remodelling.inverse_model.evaluate_inverse import (
    evaluate_predictions_with_surrogate,
    inverse_model_metrics,
)
from bone_remodelling.inverse_model.triangular_to_params_converter import (
    params_to_force_profile,
)
from bone_remodelling.parameters import ConfigurationParameters
from bone_remodelling.surrogate_model.loader import SurrogatePredictor
from bone_remodelling.surrogate_model.neural_network import SurrogateModel
from bone_remodelling.surrogate_model.surrogate_parameters import (
    SurrogateTrainParameters,
)

logger = logging.getLogger(__name__)

@dataclass
class BayesianParameters:
    """Parameters for the Bayesian inverse model."""

    device: torch.device
    learning_rate: float = 0.01
    regularization_lambda: float = 0.1
    force_dim: tuple[int, int] = (3, 10)
    param_dim: int = 3
    max_iterations: int = 20
    max_peak_height: float = 200.0

def map_inverse(
    surrogate_model: SurrogatePredictor,
    observed_density: torch.Tensor,
    bayesian_parameters: BayesianParameters,
) -> np.ndarray:
    """Perform MAP estimation to find the force vector that best explains the observed density."""
    best_loss = float("inf")
    best_force = np.zeros(bayesian_parameters.force_dim, dtype=np.float32)

    for peak_side in range(3):
        for peak_location in range(10):
            peak_height = torch.tensor(
                [bayesian_parameters.max_peak_height / 2],
                dtype=torch.float32,
                device=bayesian_parameters.device,
                requires_grad=True,
            )

            optimizer = torch.optim.LBFGS([peak_height], lr=bayesian_parameters.learning_rate)

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                force_np = params_to_force_profile(
                    peak_location, peak_side, peak_height[0],
                )
                force_tensor = torch.tensor(force_np, dtype=torch.float32, device=bayesian_parameters.device).unsqueeze(0)
                # Compute surrogate prediction
                density_pred = surrogate_model.torch_prediction(force_tensor)
                # Loss = data + prior
                data_loss = torch.mean((density_pred - observed_density) ** 2)
                prior_loss = bayesian_parameters.regularization_lambda * torch.mean(force_tensor ** 2)
                loss = data_loss + prior_loss
                loss.backward()
                return loss

            for _ in range(bayesian_parameters.max_iterations):
                optimizer.step(closure)

            # Evaluate final loss
            force_np = params_to_force_profile(peak_location, peak_side, peak_height.item())
            force_tensor = torch.tensor(force_np, dtype=torch.float32, device=bayesian_parameters.device).unsqueeze(0)
            final_pred = surrogate_model.torch_prediction(force_tensor)
            final_loss = torch.mean((final_pred - observed_density) ** 2).item()

            if final_loss < best_loss:
                best_loss = final_loss
                best_force = force_np
    return best_force

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = ConfigurationParameters().output_dir.resolve()
    surogate_path = data_path / Path("surrogate_models", "model_1_19.pth")
    train_parameters = SurrogateTrainParameters(model_path=surogate_path, device=device)
    surrogate_model = SurrogatePredictor(SurrogateModel, train_parameters)
    surrogate_model.load_model()

    raw_path = data_path / Path("raw", "triangular")
    data = ForwardDataManager(raw_path).load_directory()
    if data is None:
        raise ValueError("Failed to load data")
    _, true_forces, output_density = data
    bayesian_parameters = BayesianParameters(device=device)

    number_of_samples: int = 10
    estimated_forces: np.ndarray = np.zeros((number_of_samples, *bayesian_parameters.force_dim), dtype=np.float32)

    for i in range(number_of_samples):
        logger.info("Processing sample %d", i)
        observed_density = torch.tensor(output_density[i], device=bayesian_parameters.device, dtype=torch.float32).unsqueeze(0)
        estimated_forces[i] = map_inverse(surrogate_model, observed_density, bayesian_parameters)

    inverse_model_metrics(estimated_forces, true_forces[:number_of_samples])
    evaluate_predictions_with_surrogate(surogate_path, device, force_predictions=estimated_forces, true_densities=output_density[:number_of_samples], metric="ssim", true_forces=true_forces[:number_of_samples])
