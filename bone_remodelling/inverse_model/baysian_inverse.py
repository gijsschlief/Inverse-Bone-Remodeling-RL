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
    params_to_force_profile_torch,
)
from bone_remodelling.parameters import ConfigurationParameters
from bone_remodelling.surrogate_model.loader import SurrogatePredictor
from bone_remodelling.surrogate_model.neural_network import SurrogateModel
from bone_remodelling.surrogate_model.sanitizer import sanitize_data
from bone_remodelling.surrogate_model.splitter import splitting
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
                force_tensor = params_to_force_profile_torch(
                    peak_location, peak_side, peak_height,
                    device=bayesian_parameters.device,
                ).unsqueeze(0)

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
            force_np = params_to_force_profile_torch(peak_location, peak_side, peak_height, device=bayesian_parameters.device).detach().numpy()
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
    random_state = ConfigurationParameters().seed
    surrogate_path = data_path / Path("surrogate_models", "model_1_19.pth")
    train_parameters = SurrogateTrainParameters(model_path=surrogate_path, device=device)
    surrogate_model = SurrogatePredictor(SurrogateModel, train_parameters)
    surrogate_model.load_model()

    raw_path = data_path / Path("raw", "triangular")
    data = ForwardDataManager(raw_path).load_directory()
    if data is None:
        raise ValueError("Failed to load data")
    _, force_profiles, final_output_densities = data

    force_profiles, final_output_densities = sanitize_data(
        force_profiles,
        final_output_densities,
    )

    _, _, force_test, _, _, density_test = splitting(
        force_profiles,
        final_output_densities,
        random_state=random_state,
    )

    bayesian_parameters = BayesianParameters(device=device)

    representative_subset: int = min(10, density_test.shape[0])
    estimated_forces: np.ndarray = np.zeros((representative_subset, *bayesian_parameters.force_dim), dtype=np.float32)

    for i in range(representative_subset):
        logger.info("Processing sample %d", i)
        observed_density = torch.tensor(density_test[i], device=bayesian_parameters.device, dtype=torch.float32).unsqueeze(0)
        estimated_forces[i] = map_inverse(surrogate_model, observed_density, bayesian_parameters)

    inverse_model_metrics(estimated_forces, force_test[:representative_subset])
    evaluate_predictions_with_surrogate(surrogate_path, device, force_predictions=estimated_forces, true_densities=density_test[:representative_subset], metric="ssim", true_forces=force_test[:representative_subset])
