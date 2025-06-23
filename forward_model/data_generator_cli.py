"""Command-line interface for generating data for the forward model.

This script parses command-line arguments, initializes the TrainingDataGenerator,
and generates data in the specified mode (parallel, sequential, or edge cases).
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from fenics import LogLevel, set_log_level  # type: ignore

from forward_model.data_generator import TrainingDataGenerator  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse command line arguments and generate training data.

    It initializes the TrainingDataGenerator with the provided parameters and
    generates data in the specified mode (parallel, sequential, or edge cases).
    The generated data is saved in the specified output directory.

    Usage:
    -----
    python data_generator_cli.py [Options]

    Options:
    ------
    - `--help`: Show this help message and exit.
    - `--output_dir`: Directory to save the generated data.
    - `--num_samples`: Number of samples to generate.
    - `--initial_density_value`: Initial value to fill the density array.
    - `--x_shape`: Number of rows in the initial density array.
    - `--y_shape`: Number of columns in the initial density array.
    - `--force_max`: Maximum force magnitude.
    - `--force_count_max`: Maximum number of force applications.
    - `--batch_seed`: Random seed for reproducibility.
    - `-m` or `--mode`: Generation mode ('parallel', 'sequential', or 'edge').
    - `-v` or `--verbose`: Enable verbose output.

    """
    parser = argparse.ArgumentParser(
        description="Generate training or edge case data for the forward model.",
    )
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_dir),
        help="Directory to save output.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--initial_density_value",
        type=float,
        default=0.8,
        help="Initial density value to fill the array.",
    )
    parser.add_argument(
        "--x_shape",
        type=int,
        default=10,
        help="Number of rows in the initial density array.",
    )
    parser.add_argument(
        "--y_shape",
        type=int,
        default=10,
        help="Number of columns in the initial density array.",
    )
    parser.add_argument(
        "--force_max",
        type=int,
        default=2,
        help="Maximum force magnitude.",
    )
    parser.add_argument(
        "--force_count_max",
        type=int,
        default=7,
        help="Max number of force applications.",
    )
    parser.add_argument(
        "--batch_seed",
        type=int,
        default=np.random.randint(0, 1_000_000),
        help="Random seed.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["parallel", "sequential", "edge"],
        default="parallel",
        help="Generation mode: 'parallel', 'sequential', or 'edge'.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    args = parser.parse_args()

    if args.verbose:
        set_log_level(LogLevel.INFO)
    else:
        set_log_level(LogLevel.ERROR)

    initial_density = np.full((args.x_shape, args.y_shape), args.initial_density_value)

    generator = TrainingDataGenerator(
        output_dir=args.output_dir,
        initial_density=initial_density,
        force_max=args.force_max,
        force_count_max=args.force_count_max,
        batch_seed=args.batch_seed,
    )

    start_time = time.time()
    if args.mode == "parallel":
        generator.generate_parallel(args.num_samples)
    elif args.mode == "sequential":
        generator.generate_sequential(args.num_samples)
    elif args.mode == "edge":
        generator.generate_edge_cases(args.num_samples)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logger.info(f"Simulation completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
