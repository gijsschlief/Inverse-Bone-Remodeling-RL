"""Forward pass module for bone remodeling models."""

from abc import ABC, abstractmethod

import numpy as np

from bone_remodelling.forward_data.force_profile_generator import ForceProfileGenerator
from bone_remodelling.forward_model.main import DensitySimulation
from bone_remodelling.forward_model.parameters import SimulationParameters
from bone_remodelling.parameters import ConfigurationParameters
from bone_remodelling.surrogate_model.loader import (
    load_surrogate_models,
    predict_with_surrogates,
)
from bone_remodelling.surrogate_model.neural_network import (
    SurrogateModel,
)
from bone_remodelling.surrogate_model.surrogate_parameters import TrainParameters


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
        config: ConfigurationParameters,
        model_class: type[SurrogateModel],
        train_parameters: TrainParameters,
    ) -> None:
        """Initialize the SurrogateModel.

        Args:
        ----
            config (ConfigurationParameters): Configuration parameters for the surrogate model.
            model_class (type[SurrogateModel]): Class of the surrogate model to be loaded.
            train_parameters (TrainParameters): Training parameters for the surrogate model.

        """
        self.batch_size = config.seed
        self.density_shape = (config.mesh_top_resolution, config.mesh_side_resolution)
        self.predictors = load_surrogate_models(model_class, train_parameters)

    def forward_pass(self, force_profile: np.ndarray) -> np.ndarray:
        """Forward pass through the surrogate model.

        Args:
        ----
            force_profile (np.ndarray): The force profile applied to the bone.

        Returns:
        -------
            np.ndarray: The predicted density from the surrogate model.

        """
        x_flat = force_profile.flatten()
        x_input = x_flat[np.newaxis, :]
        return predict_with_surrogates(
            self.predictors, x_input, batch_size=self.batch_size
        )[0].reshape(self.density_shape)


class FenicsForwarder(ForwardPass):
    """Forward model for predicting bone density based on force profile."""

    def __init__(
        self,
        config: ConfigurationParameters,
        force_profile: np.ndarray,
    ) -> None:
        """Initialize the ForwardModel in fenics."""
        force_generator = ForceProfileGenerator(
            (config.force_top_resolution, config.force_side_resolution)
        )
        force_mask = force_generator.generate_force_mask()
        simulation_parameters = SimulationParameters(
            force_profile=force_profile,
            force_mask=force_mask,
            initial_density_field=np.ones(
                (config.mesh_top_resolution, config.mesh_side_resolution)
            )
            * 0.8,
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
