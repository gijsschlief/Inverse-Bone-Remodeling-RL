"""Forward model for bone remodeling simulation.

This module provides a function to calculate the bone density profile based on the force locations on the model.
#         file_path = Path(file_path)
#     if not file_path.exists():
#         logging.error(f"File {file_path} does not exist.")
#         return None
#     if file_path.suffix != ".json":
#         logging.error(f"File {file_path} is not a JSON file.")
#         return None
#     return _forward_data_load_single(file_path)
#
# Forward model for bone remodeling simulation.

This module provides a function to calculate the bone density profile based on the force locations on the model.
It uses the `DensitySimulation` class to run the simulation and can plot or save results based on parameters.
"""

import logging
from typing import Any, Optional

import numpy as np

from forward_model.density_simulation import DensitySimulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


#TODO: Has become obsolete, remove in future

def forward_model(
    force_profile: np.ndarray,
    initial_density: np.ndarray,
    time_steps: int = 100,
    dt: float = 1.0,
    parameters: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Calculate the bone density profile based on the force locations on the model.

    Args:
    ----
        force_profile (np.ndarray): Force profile matrix.
        initial_density (np.ndarray): Initial bone density matrix.
        time_steps (int): Number of time steps for the simulation.
        dt (float): Time step size.
        parameters (Dict[str, Union[str, float]]): Parameters for the forward model, including:
            - 'output_dir': Location of the data files.
            - 'min_density': Minimum bone density.
            - 'max_density': Maximum bone density.
            - 'boundary_tolerance': Boundary threshold.
            - 'plot': Whether to plot the density simulation.
            - 'save': Whether to save the simulation results.
            - 'remodeling_rate_coefficient': Coefficient for density change.
            - 'stimulus_threshold': Threshold for density change.
            - 'poisson_ratio': Poisson's ratio.
            - 'elastic_modulus_scale': Modulus of elasticity.
            - 'modulus_exponent': Exponent for density elasticity.
            - 'output_basename': Base name for output files.
            - 'file_extension': File extension for output files.
            - 'convergence_tolerance': Convergence threshold for density change.

    Returns:
    -------
        np.ndarray: Updated bone density profile after the simulation.

    """
    simulation = DensitySimulation(
        force_profile=force_profile,
        initial_density_field=initial_density,
        time_steps=time_steps,
        dt=dt,
        parameters=parameters,
    )
    simulation.run()

    # Plot density if the plot parameter is enabled
    if isinstance(parameters, dict) and parameters.get("plot", False):
        simulation.plot_density()

    # Convert the final density function to a NumPy array
    final_density = simulation.get_density()
    return final_density
