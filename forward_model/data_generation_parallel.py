import argparse
import datetime
import os
import json
from typing import Dict, Tuple
import logging
from pathlib import Path

import numpy as np
from fenics import set_log_level, LogLevel
from multiprocessing import Pool, cpu_count

from forward_model.main import forward_model
from forward_model.data_serialization import serialize_data

def _run_one_sample(args: Tuple[int, str, np.ndarray, int, float, Dict]) -> Dict:
    """
    Run a single sample of the forward model simulation with a random force profile.
    """
    i, _, initial_density, time_steps, dt, parameters, force_max, force_count_max, batch_seed = args

    force_profile = np.zeros((3, max(initial_density.shape)))
    np.random.seed(batch_seed + i)
    num_forces = np.random.randint(1, force_count_max)
    locations = np.random.choice(np.prod(force_profile.shape), num_forces, replace=False)
    for loc in locations:
        row, col = divmod(loc, force_profile.shape[1])
        force_profile[row, col] = np.random.uniform(-force_max, force_max)

    e = None
    try:
        output = forward_model(force_profile, initial_density, time_steps, dt, parameters)
    except Exception as ex:
        e = ex
        logging.error(f"Error in sample {i}: {e} | Force profile: {force_profile}")
        output = np.full(initial_density.shape, np.nan)
        
    return serialize_data(
        serial_number=i + 1,
        force_profile=force_profile,
        result=output,
        error=str(e) if isinstance(e, Exception) else None
    )

def generate_training_data(
    output_dir: str, 
    num_samples: int,
    force_max: int = 2,
    force_count_max: int = 7,
    batch_seed: int = 0
    ) -> None:
    """
    Generates training data by creating random force profiles, running a forward model,
    and saving the results to a timestamped file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    filepath = os.path.join(output_dir, f"training_batch_{batch_seed}_samples_{num_samples}_{timestamp}.json")

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

    args = [(i, output_dir, initial_density, time_steps, dt, parameters, force_max, force_count_max, batch_seed) for i in range(num_samples)]

    logging.info("Starting parallel simulation...")
    logging.info(f"Using batch seed: {batch_seed} to generate {num_samples} samples.")

    with Pool(processes=cpu_count()) as pool:
        results = []
        for idx, result in enumerate(pool.imap_unordered(_run_one_sample, args), 1):
            results.append(result)
            # Progress bar
            progress = (idx / num_samples) * 100
            logging.info(f"Progress: [{'#' * int(progress // 2)}{'.' * (50 - int(progress // 2))}] {progress:.2f}%")

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"Training data saved to {filepath}")

def main() -> None:
    """
    Main function to parse arguments and generate training data.
    """
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    parser = argparse.ArgumentParser(description="Generate training data for the forward model.")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=str(default_dir), 
        help="Directory to save the generated training data."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100, 
        help="Number of samples to generate."
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    generate_training_data(output_dir=args.output_dir, num_samples=args.num_samples, force_max=args.force_max, force_count_max=args.force_count_max, batch_seed=args.batch_seed)

if __name__ == "__main__":
    main()