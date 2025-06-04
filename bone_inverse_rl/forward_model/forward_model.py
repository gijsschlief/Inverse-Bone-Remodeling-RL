from __future__ import print_function
import numpy as np
from typing import Dict, Union
from bone_inverse_rl.forward_model.forward_model_input_test import forward_model_input_test
from bone_inverse_rl.forward_model.density_simulation import DensitySimulation

def forward_model(
    force_profile: np.ndarray,
    initial_density: np.ndarray,
    time_steps: int,
    dt: float,
    parameters: Dict[str, Union[str, float]]
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

    Returns:
        np.ndarray: Updated bone density profile after the simulation.
    """
    # Validate inputs
    error = forward_model_input_test(
        force_profile, initial_density, time_steps, dt, parameters
    )
    if error is not None:
        error_type, error_message = error
        if error_type == "ValueError":
            raise ValueError(error_message)
        elif error_type == "TypeError":
            raise TypeError(error_message)
        else:
            raise Exception(f"Unhandled error type: {error_type} - {error_message}")

    # If inputs are valid, proceed with the forward model computation

    simulation = DensitySimulation(
        force_profile=force_profile,
        initial_density=initial_density,
        time_steps=time_steps,
        dt=dt,
        parameters=parameters,
        )
    simulation.run()

    #simulation.plot_density()
    
    # Convert the final density function to a NumPy array
    final_density = simulation.get_final_density()

    return final_density

if __name__ == "__main__":
    force_profile = np.zeros((3, 40))  # Initialize an empty force profile
    force_profile[0, 0] = 0  # Set a force at location (0, 2) Top
    force_profile[0, 9] = 3  # Set a force at location (1, 5) Left
    force_profile[2, 8] = 0  # Set a force at location (2, 8) Left
    initial_density = np.full((40, 40), 0.8)  # Initial density matrix
    time_steps = 100  # Number of time steps
    dt = 1.0  # Time step size
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': 0.01,
        'rho_max': 1.74,
        'tolerance': 1E-14,
        'save': True,
    }

    final_density = forward_model(force_profile, initial_density, time_steps, dt, parameters)
    print("Force Profile:", force_profile)
    print("Final Density:", final_density)
