from __future__ import print_function
import numpy as np
from typing import Union, Tuple, Optional, Dict

def forward_model_input_test(
    force_profile: np.ndarray,
    initial_density: np.ndarray,
    time_steps: int,
    dt: Union[int, float],
    parameters: Dict[str, Union[str, float]]
) -> Optional[Union[Tuple[str, str], None]]:
    """
    This function tests if the inputs are valid for the forward model and provides error and warning methods for invalid inputs.

    Parameters:
        force_profile (np.ndarray): Force profile matrix.
        initial_density (np.ndarray): Initial bone density matrix.
        time_steps (int): Number of time steps for the simulation.
        dt (Union[int, float]): Time step size.
        file_location (str): Location of the data files.
        rho_min (Union[int, float]): Minimum bone density.
        rho_max (Union[int, float]): Maximum bone density.

    Returns:
        Optional[Tuple[str, str]]:
            - None: If all inputs are valid.
            - Tuple: Contains ('error_type', 'error_message') if any input is invalid.
    """
    file_location = parameters.get('file_location', 'bone_inverse_rl/data/raw')
    rho_min = parameters.get('rho_min', 0.01)
    rho_max = parameters.get('rho_max', 1.74)

    if not isinstance(force_profile, np.ndarray):
        return "TypeError", "force_profile must be a numpy array."
    if not isinstance(initial_density, np.ndarray):
        return "TypeError", "initial_density must be a numpy array."
    if not isinstance(time_steps, int) or time_steps <= 0:
        return "ValueError", "time_steps must be a positive integer."
    if not isinstance(dt, (int, float)) or dt <= 0:
        return "ValueError", "dt must be a positive number."
    if not isinstance(file_location, str):
        return "TypeError", "file_location must be a string."
    if not isinstance(rho_min, (int, float)):
        return "TypeError", "rho_min must be a number."
    if not isinstance(rho_max, (int, float)):
        return "TypeError", "rho_max must be a number."
    if rho_min < 0 or rho_max <= rho_min:
        return "ValueError", "rho_min must be non-negative and rho_max must be greater than rho_min."
    if force_profile.shape != initial_density.shape:
        return "ValueError", "force_profile and initial_density must have the same shape."
    if not (0 <= initial_density).all() or not (initial_density <= 1).all():
        return "ValueError", "initial_density values must be between 0 and 1."
    if not (0 <= force_profile).all():
        return "ValueError", "force_profile values must be non-negative."
    if not (0 <= dt <= 1):
        return "ValueError", "dt must be between 0 and 1."
    if not (0 <= time_steps <= 1000):
        return "ValueError", "time_steps must be between 0 and 1000."
    # If all checks pass, return None indicating no errors

    return None

if __name__ == "__main__":
    force_profile = np.random.rand(10, 10)  # Example force profile
    initial_density = np.full((10, 10), 0.8)  # Initial density matrix
    time_steps = 100  # Number of time steps
    dt = 1.0  # Time step size
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': -0.01,
        'rho_max': 1.74,
    }

    error_type, error_message = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    if error_type is not None:
        if error_type == "ValueError":
            raise ValueError(error_message)
        if error_type == "TypeError":
            raise TypeError(error_message)
    else :
        print("All inputs are valid.")