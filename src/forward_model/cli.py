"""Command line interface for running the density simulation.

This script allows users to run the density simulation with various parameters
and options for force profiles, density profiles, and output settings.
It supports command line arguments for flexibility and ease of use.

Usage:
-----
    python cli.py [options]

Options:
    --time_steps <int>                Number of time steps for the simulation.
    --dt <float>                      Time step size.
    -rho_min, --min_density <float>   Minimum bone density.
    -rho_max, --max_density <float>   Maximum bone density.
    --boundary_tolerance <float>      Tolerance for convergence criteria.
    -B, --remodeling_rate_coefficient <float>
                                      Coefficient for density change.
    -k, --stimulus_threshold <float>  Threshold for density change.
    -P, --poisson_ratio <float>       Poisson's ratio.
    -M, --elastic_modulus_scale <float>
                                      Modulus of elasticity.
    -gamma, --modulus_exponent <float> Exponent for density elasticity.
    --output_basename <str>           Base name for output files.
    --file_extension <str>            File extension for output files.
    --convergence_tolerance <float>   Convergence threshold for density change.
    --output_dir <str>                Location of the data files.
    --initial_density_value <float>   Initial density value to fill the array (default: 0.8).
    --n_rows <int>                    Number of rows in the initial density array (default: 10).
    --n_columns <int>                 Number of columns in the initial density array (default: 10).
    -f, --force <side,location,magnitude>
                                      Specify a force in the format side ('top' / 'left' / 'right'), location [int], magnitude [float].
                                      Use multiple -f or --force arguments for multiple forces.
    --save                            Save the simulation results.
    -p, --plot                        Plot the density simulation.
    -v, --verbose                     Enable verbose output (different levels available).

"""

import argparse
import logging
import time

import numpy as np
from fenics import LogLevel, set_log_level  # type: ignore

from bone_remodeling.src.forward_model.density_visualizer import plot_density_pyvista
from bone_remodeling.src.forward_model.main import DensitySimulation
from bone_remodeling.src.forward_model.parameters import SimulationParameters

logger = logging.getLogger(__name__)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the forward bone remodeling simulation.

    This function parses command line arguments and initializes the simulation.
    It sets up the force profile and parameters, runs the simulation, and logs the results.
    """
    # Data
    parser = argparse.ArgumentParser(description="Run the forward model simulation.")
    parser.add_argument(
        "--time_steps",
        type=int,
        help="Number of time steps for the simulation.",
    )
    parser.add_argument("--dt", type=float, help="Time step size.")
    parser.add_argument(
        "-rho_min",
        "--min_density",
        type=float,
        help="Minimum bone density.",
    )
    parser.add_argument(
        "-rho_max",
        "--max_density",
        type=float,
        help="Maximum bone density.",
    )
    parser.add_argument(
        "--boundary_tolerance",
        type=float,
        help="Tolerance for convergence criteria.",
    )
    parser.add_argument(
        "-B",
        "--remodeling_rate_coefficient",
        type=float,
        help="Coefficient for density change.",
    )
    parser.add_argument(
        "-k",
        "--stimulus_threshold",
        type=float,
        help="Threshold for density change.",
    )
    parser.add_argument("-P", "--poisson_ratio", type=float, help="Poisson's ratio.")
    parser.add_argument(
        "-M",
        "--elastic_modulus_scale",
        type=float,
        help="Modulus of elasticity.",
    )
    parser.add_argument(
        "-gamma",
        "--modulus_exponent",
        type=float,
        help="Exponent for density elasticity.",
    )
    parser.add_argument(
        "--output_basename",
        type=str,
        help="Base name for output files.",
    )
    parser.add_argument(
        "--file_extension",
        type=str,
        help="File extension for output files.",
    )
    parser.add_argument(
        "--convergence_tolerance",
        type=float,
        help="Convergence threshold for density change.",
    )
    parser.add_argument("--output_dir", type=str, help="Location of the data files.")

    # Density profile
    parser.add_argument(
        "--initial_density_value",
        type=float,
        default=0.8,
        help="Initial density value to fill the array.",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=10,
        help="Number of rows in the initial density array.",
    )
    parser.add_argument(
        "--n_columns",
        type=int,
        default=10,
        help="Number of columns in the initial density array.",
    )

    # Force profile
    parser.add_argument(
        "-f",
        "--force",
        action="append",
        help="Specify a force in the format side ('top' / 'left' / 'right'), location [int], magnitude [float]. Use multiple -f or --force arguments for multiple forces.",
    )
    # Actions
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the simulation results.",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot the density simulation.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    parser.add_argument(
        "-vv",
        "--very-verbose",
        action="store_true",
        help="Enable very verbose output.",
    )

    parser.add_argument(
        "-vvv",
        "--very_very_verbose",
        action="store_true",
        help="Enable very very verbose output.",
    )

    args = parser.parse_args()

    if args.plot:
        args.save = True

    if args.very_very_verbose:
        set_log_level(LogLevel.TRACE)
        args.verbose = True
        logger.info("Initializing force profile and parameters...")
    elif args.very_verbose:
        set_log_level(LogLevel.INFO)
        args.verbose = True
        logger.info("Initializing force profile and parameters...")
    elif args.verbose:
        logger.info("Initializing force profile and parameters...")
        set_log_level(LogLevel.ERROR)
    else:
        set_log_level(LogLevel.ERROR)

    initial_density = np.full((args.n_rows, args.n_columns), args.initial_density_value)

    force_profile = np.zeros(
        (3, max(args.n_rows, args.n_columns)),
    )
    if args.force:
        try:
            for force in args.force:
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
                    logger.warning(
                        f"Invalid side '{side}' specified in force argument: {force}",
                    )
        except ValueError:
            logger.warning(
                f"Invalid force argument format: {args.force}. Expected format: side,location,magnitude",
            )
    else:
        logger.warning("No forces specified. Using default force profile.")
        for i in range(max(args.n_rows, args.n_columns)):
            # Default force profile: 15 at the top
            if i < args.n_rows:
                force_profile[0, i] = 0.4 * i

    # Update simulation_parameters with command-line arguments, excluding reset, save, plot, verbose, force, and density profile
    simulation_parameters = SimulationParameters(
        force_profile=force_profile,
        initial_density_field=initial_density,
    )

    for key, value in vars(args).items():
        if (
            key
            not in {
                "reset",
                "verbose",
                "force",
                "initial_density_value",
                "n_rows",
                "n_columns",
            }
            and value is not None
            and hasattr(simulation_parameters, key)
        ):
            setattr(simulation_parameters, key, value)

    if args.verbose:
        logger.info("Parameters loaded: %s", simulation_parameters)

    logger.info("Running forward model simulation...")
    start_time = time.time()

    simulation = DensitySimulation(parameters=simulation_parameters)
    simulation.run()

    final_density = simulation.get_density()
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logger.info(f"Simulation completed in {elapsed_time:.2f} seconds.")

    # Log force profile and final density
    if args.verbose:
        logger.info("Force Profile: %s", force_profile)

    logger.info("Final Density: %s", final_density)

    # plot if enabled
    if args.plot:
        plot_density_pyvista(simulation_parameters)


if __name__ == "__main__":
    main()
