"""Contains a class that builds the evaluator gym environment."""

import numpy as np
from gymnasium import Env

from bone_remodelling.rl_model.environment import BoneRemodelingEnvironment
from bone_remodelling.rl_model.forward_pass import ForwardPass
from bone_remodelling.rl_model.parameters import RLParameters


class ValidationEnvironmentBuilder:
    """Build a model of the rl agent that runs the validation cycle."""

    def __init__(
        self,
        surrogate_forwarder: ForwardPass,
        rl_parameters: RLParameters = RLParameters(),
    ) -> None:
        """Initialize the validation environment builder."""
        self.surrogate_forwarder = surrogate_forwarder
        self.rl_parameters = rl_parameters

    def __call__(
        self,
        force_profile: np.ndarray,
        target_density: np.ndarray,
    ) -> Env:
        """Build and return a new BoneRemodellingEnvironment that contains exactly one sample (the provided force & density)."""
        forces = np.expand_dims(force_profile.astype(np.float32), axis=0)
        densities = np.expand_dims(target_density.astype(np.float32), axis=0)

        env = BoneRemodelingEnvironment(
            forwarder=self.surrogate_forwarder,
            target_densities=densities,
            target_forces=forces,
            rl_parameters=self.rl_parameters,
        )
        env.reset(seed=0)
        return env
