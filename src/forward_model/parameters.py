"""Simulation parameters for bone remodeling forward model."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimulationParameters:
    """Simulation parameters used by the forward model.

    Attributes
    ----------
        force_profile : np.ndarray
            Force profile applied to the bone, shape (3, n).
        initial_density : np.ndarray
            Initial density distribution of the bone, shape (n, n).
        time_steps : int
            Number of time steps for the simulation.
        dt : float
            Time step size.
        poisson_ratio : float
            Poisson's ratio for the material.
        elastic_modulus_scale : float
            Scale factor for the elastic modulus.
        modulus_exponent : float
            Exponent for the modulus scaling.
        remodeling_rate_coefficient : float
            Coefficient for the remodeling rate.
        stimulus_threshold : float
            Threshold for remodeling stimulus.
        min_density : float
            Minimum density value.
        max_density : float
            Maximum density value.
        convergence_tolerance : float
            Tolerance for convergence in the simulation.
        convergence_tolerance_decay : float
            Decay factor for convergence tolerance.
        convergence_after_steps : int
            Number of steps after which convergence is checked.
        convergence_steps_decay : float
            Decay factor for the number of steps for convergence checks.
        boundary_tolerance : float
            Tolerance for the boundary conditions.
        output_dir : str
            Directory to save output files.
        output_basename : str
            Base name for output files.
        output_extension : str
            File extension for output files.
        save_data : bool
            Whether to save the simulation results.

    Raises
    ------
        TypeError: If any of the parameters are of incorrect type.
        ValueError: If any of the parameters are out of valid range or conditions.

    """

    force_profile: np.ndarray  # shape (3,n)
    initial_density_field: np.ndarray  # shape (n,n)
    time_steps: int = 200
    dt: float = 1.0

    # material parameters
    poisson_ratio: float = 0.3
    elastic_modulus_scale: float = 3790.0 #100.0 new parameters match validation significantly better
    modulus_exponent: float = 3.0 #2.0

    # remodeling parameters
    remodeling_rate_coefficient: float = 1.0
    stimulus_threshold: float = 0.25
    min_density: float = 0.01
    max_density: float = 1.74

    # convergence parameters
    maximum_delta: float = 0.02
    convergence_tolerance: float = 1e-9
    convergence_tolerance_decay: float = 1.06
    convergence_after_steps: int = 10
    convergence_steps_decay: float = 0.96
    boundary_tolerance: float = 1e-14

    # I/O parameters
    output_dir: str = "data/fenics"
    output_basename: str = "density_simulation"
    output_extension: str = ".pvd"
    save_data: bool = False

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Validate the configuration parameters for the forward simulation."""
        if not isinstance(self.force_profile, np.ndarray):
            raise TypeError("force_profile must be a numpy array.")
        if not isinstance(self.initial_density_field, np.ndarray):
            raise TypeError("initial_density field must be a numpy array.")
        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            raise ValueError("dt must be a positive number.")
        if not isinstance(self.output_dir, str):
            raise TypeError("Output directory must be a string.")
        if not isinstance(self.min_density, (int, float)):
            raise TypeError("min_density must be a number.")
        if not isinstance(self.max_density, (int, float)):
            raise TypeError("max_density must be a number.")
        if self.min_density < 0 or self.max_density <= self.min_density:
            raise ValueError(
                "r_min must be non-negative and max_density must be greater than min_density.",
            )

        sides_with_forces = 3
        if self.force_profile.shape[0] != sides_with_forces or self.force_profile.shape[
            1
        ] != np.max(
            self.initial_density_field.shape,
        ):
            raise ValueError(
                "force_profile must have 3 rows and columns equal to the maximum of initial_density field dimensions.",
            )
        if np.isnan(self.force_profile).any():
            raise ValueError("force_profile contains NaN values.")
        if np.isnan(self.initial_density_field).any():
            raise ValueError("initial_density field contains NaN values.")
        if (
            not (self.min_density <= self.initial_density_field).all()
            or not (self.initial_density_field <= self.max_density).all()
        ):
            raise ValueError(
                "initial_density field values must be between min_density and max_density.",
            )
        if not (self.dt > 0):
            raise ValueError(
                "dt must be a positive number.",
            )

        large_time_step_warning = 1000
        if not (self.time_steps <= large_time_step_warning):
            logger.warning(
                "time_steps is set to a high value, which may lead to long computation times.",
            )
        if not isinstance(self.save_data, bool):
            raise TypeError("save must be a boolean value.")
        if (
            not isinstance(self.convergence_tolerance, (int, float))
            or self.convergence_tolerance <= 0
        ):
            raise ValueError("convergence_tolerance must be a positive number.")

        large_convergence_tolerance_warning = 0.1
        if self.convergence_tolerance > large_convergence_tolerance_warning:
            logger.warning("convergence tolerance is very large")
        if not isinstance(self.output_basename, str):
            raise TypeError("output_basename must be a string.")
        if not isinstance(self.output_extension, str):
            raise TypeError("file_extension must be a string.")
        if not self.output_extension.startswith("."):
            raise ValueError("file_extension must start with a dot (e.g., '.pvd').")
        if (
            not isinstance(self.convergence_after_steps, int)
            or self.convergence_after_steps <= 0
        ):
            raise ValueError("convergence_after_steps must be a positive integer.")
