"""Simulation parameters for bone remodeling forward model."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
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
        maximum_delta : float
            Maximum allowable change in density per time step.
        krylov_solver_tolerance : float
            Tolerance for the Krylov solver.
        krylov_solver_iterations : int
            Maximum iterations for the Krylov solver.
        linear_solver : str
            Type of linear solver to use.
        preconditioner : str
            Type of preconditioner to use.

    Raises
    ------
        TypeError: If any of the parameters are of incorrect type.
        ValueError: If any of the parameters are out of valid range or conditions.

    """

    force_profile: np.ndarray  # shape (3,n)
    force_mask: np.ndarray  # bool of shape (3,n) to define the resolution of applied forces
    initial_density_field: np.ndarray  # shape (n,n)

    # material parameters based on Weinans et al. 1992
    poisson_ratio: float = 0.3
    elastic_modulus_scale: float = 3790.0
    modulus_exponent: float = 3.0
    remodeling_rate_coefficient: float = 1.0
    stimulus_threshold: float = 0.25
    min_density: float = 0.01
    max_density: float = 1.74
    dt: float = 1.0

    # Max remodelling rate based on viscous remodeling assumption
    maximum_delta: float = 0.02

    # FEM element orders (low orders result in checkerboarding and mesh-dependency)
    displacement_element_order: int = 3
    density_element_order: int = 0

    # convergence parameters based on parameter sweep experiment
    time_steps: int = 250
    convergence_tolerance_decay: float = 1.06
    convergence_after_steps: int = 10
    convergence_steps_decay: float = 1

    # standard value for Krylov solver
    linear_solver: str = "default"
    preconditioner: str = "hypre_amg"
    krylov_solver_iterations: int = 1000

    # Numerical parameters based on machine precision float 64-bit = 2.22e-16 < boundary_tolerance < krylov_solver_tolerance < convergence_tolerance
    convergence_tolerance: float = 1e-9
    krylov_solver_tolerance: float = 1e-10
    boundary_tolerance: float = 1e-14

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Validate the configuration parameters for the forward simulation."""
        if not isinstance(self.force_profile, np.ndarray):
            raise TypeError("force_profile must be a numpy array.")
        if not isinstance(self.initial_density_field, np.ndarray):
            raise TypeError("initial_density field must be a numpy array.")
        if not isinstance(self.force_mask, np.ndarray):
            raise TypeError("force_mask must be a numpy array.")
        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            raise ValueError("dt must be a positive number.")
        if not isinstance(self.min_density, (int, float)):
            raise TypeError("min_density must be a number.")
        if not isinstance(self.max_density, (int, float)):
            raise TypeError("max_density must be a number.")
        if self.min_density < 0 or self.max_density <= self.min_density:
            raise ValueError(
                "r_min must be non-negative and max_density must be greater than min_density.",
            )

        sides_with_forces = 3
        if self.force_profile.shape[0] != sides_with_forces:
            raise ValueError(
                "force_profile must have 3 rows.",
            )
        if np.isnan(self.force_profile).any():
            raise ValueError("force_profile contains NaN values.")
        if np.issubdtype(self.force_mask.dtype, np.bool_) is False:
            raise ValueError("force_mask must be a boolean array.")
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
        if (
            not isinstance(self.convergence_tolerance, (int, float))
            or self.convergence_tolerance <= 0
        ):
            raise ValueError("convergence_tolerance must be a positive number.")

        large_convergence_tolerance_warning = 0.1
        if self.convergence_tolerance > large_convergence_tolerance_warning:
            logger.warning("convergence tolerance is very large")
        if (
            not isinstance(self.convergence_after_steps, int)
            or self.convergence_after_steps <= 0
        ):
            raise ValueError("convergence_after_steps must be a positive integer.")
