import logging
from typing import Any, Dict, Union
from pathlib import Path

import numpy as np

from forward_model.density_simulation import DensitySimulation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def forward_model(
    force_profile: np.ndarray,
    initial_density: np.ndarray,
    time_steps: int = 100,
    dt: float = 1.0,
    parameters: Dict[str, Union[str, float]] = {}
) -> np.ndarray:
    """
    Calculate the bone density profile based on the force locations on the model.

    Parameters:
        force_profile (np.ndarray): Force profile matrix.
        initial_density (np.ndarray): Initial bone density matrix.
        time_steps (int): Number of time steps for the simulation.
        dt (float): Time step size.
        parameters (Dict[str, Union[str, float]]): Parameters for the forward model, including:
            - 'file_location': Location of the data files.
            - 'rho_min': Minimum bone density.
            - 'rho_max': Maximum bone density.
            - 'tolerance': Tolerance for convergence criteria.
            - 'plot': Whether to plot the density simulation.
            - 'save': Whether to save the simulation results.
            - 'B': Coefficient for density change.
            - 'k': Threshold for density change.
            - 'nu': Poisson's ratio.
            - 'M': Modulus of elasticity.
            - 'gamma': Exponent for density elasticity.
            - 'file_name': Base name for output files.
            - 'file_extension': File extension for output files.
            - 'convergence_eps': Convergence threshold for density change.

    Returns:
        np.ndarray: Updated bone density profile after the simulation.
    """
    simulation = DensitySimulation(
        force_profile=force_profile,
        initial_density=initial_density,
        time_steps=time_steps,
        dt=dt,
        parameters=parameters,
        )
    simulation.run()

    # Plot density if the plot parameter is enabled
    if parameters.get('plot', False):
        simulation.plot_density()
    
    # Convert the final density function to a NumPy array
    final_density = simulation.get_final_density()
    return final_density