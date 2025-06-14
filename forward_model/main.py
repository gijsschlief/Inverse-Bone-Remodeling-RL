import argparse
import logging
from typing import TypedDict
from pathlib import Path
import json

import numpy as np
from fenics import set_log_level, LogLevel

from forward_model.input_tester import forward_model_input_test
from forward_model.density_simulation import DensitySimulation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ForwardModelParameters(TypedDict):
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fenics"
    file_location: str = str(default_dir)
    rho_min: float = 0.01
    rho_max: float = 1.74
    tolerance: float = 1E-14
    save: bool = False
    plot: bool = False
    B: float = 0.1
    k: float = 0.01
    nu: float = 0.3
    M: float = 1.0
    gamma: float = 2.0
    file_name: str = "density_simulation"
    file_extension: str = ".pvd"
    convergence_eps: float = 1E-6

def forward_model(
    force_profile: np.ndarray,
    initial_density: np.ndarray,
    time_steps: int = 100,
    dt: float = 1.0,
    parameters: ForwardModelParameters = None
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
    if parameters is None:
        default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fenics"
        parameters = ForwardModelParameters(
            file_location=str(default_dir),
            rho_min=0.01,
            rho_max=1.74,
            tolerance=1E-14,
            save=False,
            plot=False,
            B=0.1,
            k=0.01,
            nu=0.3,
            M=1.0,
            gamma=2.0,
            file_name="density_simulation",
            file_extension=".pvd",
            convergence_eps=1E-6
        )

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

    # Plot density if the plot parameter is enabled
    if parameters.get('plot', False):
        simulation.plot_density()
    
    # Convert the final density function to a NumPy array
    final_density = simulation.get_final_density()

    return final_density

def main() -> None:
    """
    Main function to run the forward model simulation.
    This function parses command line arguments and initializes the simulation.
    It sets up the force profile and parameters, runs the simulation, and logs the results.
    """
    parser = argparse.ArgumentParser(description="Run the forward model simulation.")
    parser.add_argument("--time_steps", type=int, help="Number of time steps for the simulation.")
    parser.add_argument("--dt", type=float, help="Time step size.")
    parser.add_argument("--rho_min", type=float, help="Minimum bone density.")
    parser.add_argument("--rho_max", type=float, help="Maximum bone density.")
    parser.add_argument("--tolerance", type=float, help="Tolerance for convergence criteria.")
    parser.add_argument("--B", type=float, help="Coefficient for density change.")
    parser.add_argument("--k", type=float, help="Threshold for density change.")
    parser.add_argument("--nu", type=float, help="Poisson's ratio.")
    parser.add_argument("--M", type=float, help="Modulus of elasticity.")
    parser.add_argument("--gamma", type=float, help="Exponent for density elasticity.")
    parser.add_argument("--file_name", type=str, help="Base name for output files.")
    parser.add_argument("--file_extension", type=str, help="File extension for output files.")
    parser.add_argument("--save", action="store_true", help="Save the simulation results.")
    parser.add_argument("--plot", action="store_true", help="Plot the density simulation.")
    parser.add_argument("--convergence_eps", type=float, help="Convergence threshold for density change.")
    parser.add_argument("--file_location", type=str, help="Location of the data files.")
    parser.add_argument("--reset", action="store_true", help="Reset parameters to default by deleting parameters.json.")

    args = parser.parse_args()

    parameters_file = Path(__file__).resolve().parent / "parameters.json"

    if args.reset:
        if parameters_file.exists():
            parameters_file.unlink()  # Delete the parameters.json file
            logging.info(f"Parameters reset to default by deleting {parameters_file}.")
        else:
            logging.info("No parameters.json file found to reset.")
        return

    set_log_level(LogLevel.ERROR)  # Suppress FEniCS log messages
    logging.info("Initializing force profile and parameters...")
    force_profile = np.zeros((3, 40))  # Initialize an empty force profile
    force_profile[0, 0] = 3  # Set a force at location (0, 2) Top
    force_profile[0, 39] = 3  # Set a force at location (1, 5) Left
    force_profile[2, 8] = 0  # Set a force at location (2, 8) Left
    initial_density = np.full((40, 40), 0.8)  # Initial density matrix

    # Load existing parameters from JSON file if it exists
    if parameters_file.exists():
        with open(parameters_file, "r") as f:
            parameters = json.load(f)
    else:
        parameters = ForwardModelParameters()

    # Update parameters with command-line arguments
    for key, value in vars(args).items():
        if value is not None:
            parameters[key] = value

    # Save updated parameters to the JSON file
    with open(parameters_file, "w") as f:
        json.dump(parameters, f, indent=4)

    logging.info(f"Parameters saved to {parameters_file}")

    logging.info("Running forward model simulation...")
    final_density = forward_model(force_profile, initial_density, parameters.get('time_steps', 100), parameters.get('dt', 1.0), parameters)
    logging.info("Simulation completed.")
    logging.info("Force Profile: %s", force_profile)
    logging.info("Final Density: %s", final_density)

if __name__ == "__main__":
    main()