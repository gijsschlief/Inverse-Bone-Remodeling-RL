"""Bayesian inverse model for bone remodeling."""

import argparse
import logging
from dataclasses import dataclass, field
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
    config_parameters: ConfigurationParameters
    learning_rate: float = 1.0
    regularization_lambda: float = 0.1
    param_dim: int = 3
    max_iterations: int = 20
    max_peak_height: float = 200.0
    force_dim: tuple[int, int] = field(default_factory=lambda: (3, 0))

    def __post_init__(self) -> None:
        """Initialize force_dim based on config_parameters."""
        self.force_dim = (
            3,
            max(
                self.config_parameters.force_top_resolution,
                self.config_parameters.force_side_resolution,
            ),
        )


def map_inverse(
    surrogate_model: SurrogatePredictor,
    observed_density: torch.Tensor,
    bayesian_parameters: BayesianParameters,
) -> np.ndarray:
    """Perform MAP estimation to find the force vector that best explains the observed density."""
    best_loss = float("inf")
    best_force = np.zeros(bayesian_parameters.force_dim, dtype=np.float32)

    for peak_side in range(3):
        for peak_location in range(bayesian_parameters.force_dim[1]):
            peak_height = torch.tensor(
                [bayesian_parameters.max_peak_height / 2],
                dtype=torch.float32,
                device=bayesian_parameters.device,
                requires_grad=True,
            )

            optimiser = torch.optim.LBFGS(
                [peak_height],
                lr=bayesian_parameters.learning_rate,
                max_iter=bayesian_parameters.max_iterations,
            )
            params = (peak_location, peak_side, peak_height)
            optimiser.step(
                lambda params=params, optimiser=optimiser: closure(
                    surrogate_model,
                    observed_density,
                    bayesian_parameters,
                    params,
                    optimiser,
                ),
            )
            # Evaluate final loss
            with torch.no_grad():
                final_force_tensor = params_to_force_profile_torch(
                    peak_location,
                    peak_side,
                    peak_height,
                    bayesian_parameters.force_dim[1],
                    device=bayesian_parameters.device,
                )
                final_pred = surrogate_model.torch_prediction(
                    final_force_tensor.unsqueeze(0)
                )
                final_loss = torch.mean((final_pred - observed_density) ** 2).item()

                if final_loss < best_loss:
                    best_loss = final_loss
                    best_force = final_force_tensor.cpu().numpy()
    return best_force


def closure(
    surrogate_model: SurrogatePredictor,
    observed_density: torch.Tensor,
    bayesian_parameters: BayesianParameters,
    params: tuple[int, int, torch.Tensor],
    optimiser: torch.optim.LBFGS,
) -> torch.Tensor:
    """Closure function for LBFGS optimization."""
    location, side, peak_height = params
    optimiser.zero_grad()
    force_tensor = params_to_force_profile_torch(
        location,
        side,
        peak_height,
        bayesian_parameters.force_dim[1],
        device=bayesian_parameters.device,
    ).unsqueeze(0)
    density_pred = surrogate_model.torch_prediction(force_tensor)
    data_loss = torch.mean((density_pred - observed_density) ** 2)
    prior_loss = bayesian_parameters.regularization_lambda * torch.mean(
        force_tensor**2,
    )
    loss = data_loss + prior_loss
    loss.backward()
    return loss


def evaluate_bayesian(
    configuration_parameters: ConfigurationParameters,
    device: torch.device,
    surrogate_path: Path,
    sample_count: int | None = None,
) -> None:
    """Evaluate the Bayesian inverse model on the test set."""
    # This function is currently integrated into the cli for simplicity, but can be separated if needed.
    train_parameters = SurrogateTrainParameters(
        model_path=surrogate_path,
        device=device,
    )
    surrogate_model = SurrogatePredictor(SurrogateModel, train_parameters)
    surrogate_model.load_model()

    raw_path = configuration_parameters.output_dir / Path("raw", "triangular")
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
        random_state=configuration_parameters.seed,
    )

    bayesian_parameters = BayesianParameters(
        device=device, config_parameters=configuration_parameters
    )

    if sample_count is None:
        sample_count = len(density_test)
        logger.warning(
            "No sample count provided for Bayesian evaluation, using entire test set (%d samples)",
            sample_count,
        )
    estimated_forces: np.ndarray = np.zeros(
        (sample_count, *bayesian_parameters.force_dim),
        dtype=np.float32,
    )

    for i in range(sample_count):
        logger.info("Processing sample %d", i)
        observed_density = torch.tensor(
            density_test[i],
            device=bayesian_parameters.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        estimated_forces[i] = map_inverse(
            surrogate_model,
            observed_density,
            bayesian_parameters,
        )

    inverse_model_metrics(estimated_forces, force_test[:sample_count])
    evaluate_predictions_with_surrogate(
        surrogate_path,
        device,
        force_predictions=estimated_forces,
        true_densities=density_test[:sample_count],
        metric="ssim",
        true_forces=force_test[:sample_count],
    )


def cli(
    configuration_parameters: ConfigurationParameters,
    remaining_args: list[str],
) -> None:
    """CLI entry point for the Bayesian inverse model."""
    parser = argparse.ArgumentParser(description="Run the Bayesian inverse model.")
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples for inverse Bayesian evaluation.",
    )
    parser.add_argument(
        "--surrogate_model",
        type=str,
        help="Path to the surrogate model for Bayesian evaluation.",
    )
    args = parser.parse_args(remaining_args)
    if not args.surrogate_model:
        args.surrogate_model = "model_1.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate_path = configuration_parameters.output_dir / Path(
        "surrogate_models",
        args.surrogate_model,
    )
    evaluate_bayesian(
        configuration_parameters,
        device,
        surrogate_path,
        sample_count=args.samples,
    )


if __name__ == "__main__":
    # Developer convenience entry point.
    # For reproducible runs, use the unified CLI (main.py).
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ConfigurationParameters()
    surrogate_path = config.output_dir.resolve() / Path(
        "surrogate_models",
        "model_1_19.pth",
    )
    evaluate_bayesian(
        ConfigurationParameters(),
        device,
        surrogate_path,
        sample_count=10,
    )
