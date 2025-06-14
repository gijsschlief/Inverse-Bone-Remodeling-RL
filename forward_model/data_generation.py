import argparse
import datetime
import os
import json
import logging
from pathlib import Path

import numpy as np
from fenics import set_log_level, LogLevel

from forward_model.main import forward_model
from forward_model.data_serialization import serialize_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_training_data(
    output_dir: str, 
    num_samples: int,
    force_max: int,
    force_count_max: int,
    batch_seed: int
) -> None:

    """
    Generates training data by creating random force profiles, running a forward model,
    and saving the results to a timestamped file.

    Args:
        output_dir (str): Directory to save the output file.
        num_samples (int): Number of random force profiles to generate.
        force_max (int): Maximum force applied in the force profile.
        force_count_max (int): Maximum number of forces applied in the force profile.
        batch_seed (int): Seed for random number generation to ensure reproducibility.

    Returns:
        None: The function saves the training data to a JSON file in the specified directory.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamped filename
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    filepath = os.path.join(output_dir, f"training_batch_{batch_seed}_samples_{num_samples}_{timestamp}.json")

    # Define parameters for the forward model
    initial_density = np.full((10, 10), 0.8) 
    time_steps = 100
    dt = 1.0 
    parameters = {}

    # Read parameters from parameters.json
    parameters_file = Path(__file__).resolve().parent / "parameters.json"
    if parameters_file.exists():
        with open(parameters_file, 'r') as f:
            loaded_parameters = json.load(f)
            time_steps = loaded_parameters.get('time_steps', time_steps)
            dt = loaded_parameters.get('dt', dt)
            parameters.update(loaded_parameters)
    else:
        logging.warning(f"parameters.json not found in {parameters_file}. Using default values.")

    logging.info("Starting simulation...")
    logging.info(f"Using batch seed: {batch_seed} to generate {num_samples} samples.")

    # Generate random force profiles and collect results
    data = []
    for i in range(num_samples):
        np.random.seed(batch_seed + i)
        force_profile = np.zeros((3, max(initial_density.shape)))
        num_forces = np.random.randint(1, force_count_max)  # Random number of forces between 1 and force_count_max

        # Randomly select three unique locations on the side of the density matrix
        locations = np.random.choice(np.prod(force_profile.shape), num_forces, replace=False)
        for loc in locations:
            row, col = divmod(loc, force_profile.shape[1])
            force_profile[row, col] = np.random.uniform(-force_max, force_max)
            force_profile[row, col] = np.random.uniform(-force_max, force_max)  # Use force_max from function arguments
        e = None  # Initialize e to ensure it is always defined
        try:
            output = forward_model(force_profile, initial_density, time_steps, dt, parameters)
        except Exception as ex:
            e = ex
            logging.error(f"Error in sample {i}: {e} | Force profile: {force_profile}")
            output = np.full(initial_density.shape, np.nan)
        
        data_point = serialize_data(
            force_profile=force_profile, 
            result=output, 
            serial_number=i + 1,
            error=str(e) if isinstance(e, Exception) else None
        )
        data.append(data_point)

        # Save the collected data to a JSON file
        if (i + 1) % 100 == 0 or (i + 1) == num_samples:
            with open(filepath, 'w') as json_file:
                json.dump(data, json_file, indent=4)

        # Measure and display average time for samples 2 to 12
        if i == 1:
            start_time = datetime.datetime.now()
        elif i == 11:
            end_time = datetime.datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            avg_time_per_sample = elapsed_time / 10  # 10 samples (1 to 11 inclusive)
            logging.info(f"Average time per sample (1-11): {avg_time_per_sample:.4f} seconds")

        # Display progress bar
        progress = (i / num_samples) * 100
        logging.info(f"Progress: [{'#' * int(progress // 2)}{'.' * (50 - int(progress // 2))}] {progress:.2f}%")

    logging.info(f"Training data saved to {filepath}")
    return None

def main() -> None:
    """
    Main function to parse command line arguments and generate training data.
    It sets up the argument parser, suppresses FEniCS log messages, and calls the data generation function.
    """
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    parser = argparse.ArgumentParser(description="Generate training data for the forward model.")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=str(default_dir), 
        help="Directory to save the output file."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10, 
        help="Number of random force profiles to generate."
    )
    parser.add_argument(
        "--force_max", 
        type=int, 
        default=2, 
        help="Maximum force applied in the force profile."
    )
    parser.add_argument(
        "--force_count_max", 
        type=int, 
        default=7, 
        help="Maximum number of forces applied in the force profile."
    )
    parser.add_argument(
        "--batch_seed", 
        type=int, 
        default=np.random.randint(0, 1_000_000), 
        help="Seed for random number generation to ensure reproducibility."
    )
    args = parser.parse_args()

    set_log_level(LogLevel.ERROR)  # Suppress FEniCS log messages
    generate_training_data(output_dir=args.output_dir, num_samples=args.num_samples, force_max=args.force_max, force_count_max=args.force_count_max, batch_seed=args.batch_seed)
    return

if __name__ == "__main__":
    main()