"""Simulation parameters for bone remodeling forward model."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for the bone remodeling forward model simulation.

    Attributes
    ----------
        force_profile: np.ndarray: Force profile applied to the bone.
        initial_density: np.ndarray: Initial density distribution of the bone.
        time_steps: int: Number of time steps for the simulation.
        dt: float: Time step size.
        poisson_ratio: float: Poisson's ratio for the material.
        elastic_modulus_scale: float: Scale factor for elastic modulus.
        modulus_exponent: float: Exponent for modulus scaling.
        remod_rate_coeff: float: Coefficient for remodeling rate.
        stimulus_threshold: float: Threshold for stimulus activation.
        min_density: float: Minimum allowable density.
        max_density: float: Maximum allowable density.
        convergence_tol: float: Tolerance for convergence in remodeling.
        convergence_steps: int: Number of steps to check for convergence.
        output_dir: str: Directory to save output data.
        output_basename: str: Base name for output files.
        save: bool: Whether to save the output data.

    """

    force_profile: np.ndarray      # shape (3,n)
    initial_density: np.ndarray    # shape (n,n)
    time_steps: int = 100
    dt: float = 1.0

    # material parameters
    poisson_ratio: float = 0.3
    elastic_modulus_scale: float = 100.0
    modulus_exponent: float = 2.0

    # remodeling parameters
    remod_rate_coeff: float = 1.0
    stimulus_threshold: float = 0.25
    min_density: float = 0.01
    max_density: float = 1.74
    convergence_tol: float = 1e-6
    convergence_steps: int = 1

    # I/O parameters
    output_dir: str = "data/fenics"
    output_basename: str = "density"
    save: bool = False