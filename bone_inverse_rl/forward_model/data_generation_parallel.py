import argparse
import datetime
import os
import json
from typing import Dict, Tuple
import logging

import numpy as np
from fenics import set_log_level, LogLevel
from multiprocessing import Pool, cpu_count

from Thesis_code.bone_inverse_rl.forward_model.main import forward_model
from Thesis_code.bone_inverse_rl.forward_model.data_serialization import serialize_data

def _run_one_sample(args: Tuple[int, str, np.ndarray, int, float, Dict]) -> Dict:
    """
    Run a single sample of the forward model simulation with a random force profile.
    """
    i, output_dir, initial_density, time_steps, dt, parameters = args

    force_profile = np.zeros((3, max(initial_density.shape)))
    force_max = 2
    np.random.seed(i)  # Ensure reproducibility for each sample
    num_forces = np.random.randint(1, 7)
    locations = np.random.choice(np.prod(force_profile.shape), num_forces, replace=False)
    for loc in locations:
        row, col = divmod(loc, force_profile.shape[1])
        force_profile[row, col] = np.random.uniform(-force_max, force_max)

    e = None  # Initialize e to ensure it is always defined
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
    num_samples: int = 100
) -> None:
    """
    Generates training data by creating random force profiles, running a forward model,
    and saving the results to a timestamped file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"training_data_{timestamp}.json")

    initial_density = np.full((10, 10), 0.8)
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': output_dir,
        'rho_min': 0.01,
        'rho_max': 1.74,
        'save': False,
        'plot': False
    }

    args = [(i, output_dir, initial_density, time_steps, dt, parameters) for i in range(num_samples)]

    logging.info("Starting parallel simulation...")

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
    parser = argparse.ArgumentParser(description="Generate training data for the forward model.")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/gijs/Desktop/Thesis/data/raw", 
        help="Directory to save the generated training data."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100, 
        help="Number of samples to generate."
    )
    args = parser.parse_args()

    set_log_level(LogLevel.ERROR)  # Suppress FEniCS log messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    generate_training_data(output_dir=args.output_dir, num_samples=args.num_samples)

if __name__ == "__main__":
    main()