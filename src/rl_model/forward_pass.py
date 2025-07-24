"""Forward pass module for bone remodeling models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from bone_remodeling.src.forward_model.main import DensitySimulation
from bone_remodeling.src.forward_model.parameters import SimulationParameters
from bone_remodeling.src.surrogate_model.ensemble import (
    load_ensemble_models,
    predict_with_ensemble,
)
from bone_remodeling.src.surrogate_model.loader import (
    SurrogateModelLoader,
    load_surrogate_model,
)
from bone_remodeling.src.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)
from bone_remodeling.src.surrogate_model.normalizor import (
    normalize_data,
    unnormalize_data,
)


class ForwardPass(ABC):
    """Abstract base class for forward pass models."""

    @abstractmethod
    def forward_pass(self, force_profile: np.ndarray) -> np.ndarray:
        """Forward pass through the model.

        Args:
        ----
            force_profile (np.ndarray): The force profile applied to the bone.

        Returns:
        -------
            np.ndarray: The predicted density from the surrogate model.

        """
        pass


class SurrogateForwarder(ForwardPass):
    """Surrogate model for predicting bone density based on force profile."""

    def __init__(
        self,
        surrogate_model_path: Path,
        density_shape: tuple[int, ...],
        model_class: type[SurrogateModel],
    ) -> None:
        """Initialize the SurrogateModel.

        Args:
        ----
            surrogate_model_path (Path): Path to the surrogate model file.
            density_shape (tuple[int, ...]): Shape of the predicted density output.
            model_class (type[SurrogateModel]): Class of the surrogate model to be loaded.

        """
        surrogate_model_and_normalization_params = load_surrogate_model(
            surrogate_model_path,
            model_class,
        )
        assert (
            surrogate_model_and_normalization_params is not None
        ), f"Failed to load surrogate model from {surrogate_model_path}"

        (
            self.surrogate_model,
            self.x_mean,
            self.x_std,
            self.y_mean,
            self.y_std,
        ) = surrogate_model_and_normalization_params
        if self.surrogate_model is None:
            raise ValueError(
                f"Surrogate model could not be loaded from {surrogate_model_path}. Please check the file path and model type.",
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.surrogate_model.to(self.device).eval()
        self.surrogate_model = cast(Module, self.surrogate_model)
        self.density_shape = density_shape

    def _normalize_data(
        self, force_profile: np.ndarray,
    ) -> np.ndarray:
        """Normalize the force profile data.

        Args:
        ----
            force_profile (np.ndarray): The force profile applied to the bone.

        Returns:
        -------
            np.ndarray: Normalized force profile.

        """
        return normalize_data(
            force_profile,
            self.x_mean,
            self.x_std,
        )[0]

    def _unnormalize_data(
        self, density: Tensor,
    ) -> Tensor:
        """Unnormalize the density data.

        Args:
        ----
            density (np.ndarray): The density to unnormalize.

        Returns:
        -------
            np.ndarray: Unnormalized density.

        """
        return unnormalize_data(
            density,
            self.y_mean,
            self.y_std,
        )[0]

    def forward_pass(self, force_profile: np.ndarray) -> np.ndarray:
        """Forward pass through the surrogate model.

        Args:
        ----
            force_profile (np.ndarray): The force profile applied to the bone.

        Returns:
        -------
            np.ndarray: The predicted density from the surrogate model.

        """
        force_profile = self._normalize_data(force_profile)

        with torch.no_grad():
            force_profile_tensor = torch.from_numpy(
                force_profile.reshape(1, -1).astype(np.float32),
            ).to(self.device)
        density_tensor = self.surrogate_model(force_profile_tensor)

        density_tensor = self._unnormalize_data(density_tensor)
        surrogate_density: np.ndarray = density_tensor.detach().cpu().numpy()
        return surrogate_density.reshape(self.density_shape)


class EnsembleForwarder(ForwardPass):
    """Ensemble model for predicting bone density using multiple surrogate models."""

    def __init__(
        self,
        model_paths: list[Path],
        model_class: type[SurrogateModel],
        model_loader: type[SurrogateModelLoader],
    ) -> None:
        """Initialize the EnsembleModel with a list of surrogate models."""
        models, x_means, x_stds, y_means, y_stds = load_ensemble_models(
            model_paths, model_class, model_loader,
        )
        self.models = models
        self.x_means = x_means
        self.x_stds = x_stds
        self.y_means = y_means
        self.y_stds = y_stds

    def forward_pass(self, force_profile: np.ndarray) -> np.ndarray:
        """Forward pass through the ensemble model.

        Args:
        ----
            force_profile (np.ndarray): The force profile applied to the bone.

        Returns:
        -------
            np.ndarray: The predicted density from the ensemble model.

        """
        return predict_with_ensemble(
            self.models,
            force_profile,
            (self.x_means, self.x_stds),
            (self.y_means, self.y_stds),
        )[0].squeeze()


class FenicsForwarder(ForwardPass):
    """Forward model for predicting bone density based on force profile."""

    def __init__(
        self, force_profile: np.ndarray, initial_density_field: np.ndarray,
    ) -> None:
        """Initialize the ForwardModel in fenics."""
        simulation_parameters = SimulationParameters(
            force_profile=force_profile, initial_density_field=initial_density_field,
        )
        self.density_simulation = DensitySimulation(parameters=simulation_parameters)

    def forward_pass(self, force_profile: np.ndarray) -> np.ndarray:
        """Forward pass through the forward model.

        Args:
        ----
            force_profile (np.ndarray): The force profile applied to the bone.

        Returns:
        -------
            np.ndarray: The predicted density from the forward model.

        """
        self.density_simulation.reset()
        self.density_simulation.update_force_profile(force_profile)
        self.density_simulation.run()
        return self.density_simulation.get_density()
