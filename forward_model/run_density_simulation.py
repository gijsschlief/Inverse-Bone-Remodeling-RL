import argparse
import logging
import json
from pathlib import Path

import numpy as np
from fenics import set_log_level, LogLevel

from forward_model.main import forward_model

def setup_logging(verbose: bool) -> None:
    """
    Set up logging based on verbosity.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')

def main() -> None:
    """
    Main function to run the forward model simulation.
    This function parses command line arguments and initializes the simulation.
    It sets up the force profile and parameters, runs the simulation, and logs the results.
    """
    # Data
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
    parser.add_argument("--convergence_eps", type=float, help="Convergence threshold for density change.")
    parser.add_argument("--file_location", type=str, help="Location of the data files.")

    # Density profile
    parser.add_argument("--initial_density_value", type=float, default=0.8, help="Initial density value to fill the array.")
    parser.add_argument("--x_shape", type=int, default=10, help="Number of rows in the initial density array.")
    parser.add_argument("--y_shape", type=int, default=10, help="Number of columns in the initial density array.")

    # Force profile
    parser.add_argument("-f","--force", action="append", help="Specify a force in the format side ('top' / 'left' / 'right'), location [int], magnitude [float]. Use multiple -f or --force arguments for multiple forces.")
    # Actions
    parser.add_argument("--reset", action="store_true", help="Reset parameters to default by deleting parameters.json.")
    parser.add_argument("--save", action="store_true", help="Save the simulation results.")
    parser.add_argument("-p","--plot", action="store_true", help="Plot the density simulation.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    # Set saving to true if plotting is enabled
    if args.plot:
        args.save = True

    setup_logging(args.verbose)

    parameters_file = Path(__file__).resolve().parent / "parameters.json"

    if args.reset:
        if parameters_file.exists():
            parameters_file.unlink()  # Delete the parameters.json file
            logging.info(f"Parameters reset to default by deleting {parameters_file}.")
        else:
            logging.info("No parameters.json file found to reset.")
        return

    set_log_level(LogLevel.ERROR)  # Suppress FEniCS log messages
    if args.verbose:
        logging.info("Initializing force profile and parameters...")
        set_log_level(LogLevel.INFO)  # Set FEniCS log level to INFO for verbose output

    initial_density = np.full((args.x_shape, args.y_shape), args.initial_density_value)

    # Initialize force profile based on command-line arguments
    force_profile = np.zeros((3, max(args.x_shape, args.y_shape)))  # Initialize an empty force profile
    if args.force:
        for force in args.force:
            try:
                side, location, magnitude = force.split(",")
                side = side.strip().lower()
                location = int(location.strip())
                magnitude = float(magnitude.strip())

                if side == "top":
                    force_profile[0, location] = magnitude
                elif side == "right":
                    force_profile[1, location] = magnitude
                elif side == "left":
                    force_profile[2, location] = magnitude
                else:
                    logging.warning(f"Invalid side '{side}' specified in force argument: {force}")
            except ValueError:
                logging.warning(f"Invalid force argument format: {force}. Expected format: side,location,magnitude")
    else:
        logging.warning("No forces specified. Using default force profile.")
        force_profile[2, 2] = 3


    # Load existing parameters from JSON file if it exists
    if parameters_file.exists():
        with open(parameters_file, "r") as f:
            parameters = json.load(f)
    else:
        parameters = {}

    # Update parameters with command-line arguments, excluding reset, save, plot, verbose, force, and density profile
    for key, value in vars(args).items():
        if key not in {"reset", "save", "plot", "verbose", "force", "initial_density_value", "x_shape", "y_shape"} and value is not None:
            parameters[key] = value

    # Save updated parameters to the JSON file
    with open(parameters_file, "w") as f:
        json.dump(parameters, f, indent=4)

    # Add save and plot to parameters
    parameters['save'] = args.save
    parameters['plot'] = args.plot

    if args.verbose:
        logging.info("Parameters loaded: %s", parameters)
        logging.info(f"Parameters saved to {parameters_file}")

    logging.info("Running forward model simulation...")
    final_density = forward_model(force_profile, initial_density, parameters.get('time_steps', 100), parameters.get('dt', 1.0), parameters)
    logging.info("Simulation completed.")

    # Log less information if verbose is not enabled
    if args.verbose:
        logging.info("Force Profile: %s", force_profile)
    
    logging.info("Final Density: %s", final_density)

if __name__ == "__main__":
    main()