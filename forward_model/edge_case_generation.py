import argparse
import datetime
import os
import json
from typing import Dict, Tuple
import logging

import numpy as np
from fenics import set_log_level, LogLevel
from multiprocessing import Pool, cpu_count

from forward_model.main import forward_model
from forward_model.data_serialization import serialize_data

def _generate_edge_case_force_profiles(initial_density_shape: Tuple[int, int], num_cases: int = 2500) -> np.ndarray:
    """
    Generate a large number of edge case force profiles to stress-test the model.
    """
    edge_cases = []

    force_max = 2

    for i in range(num_cases):
        if i % 5 == 0:
            # Case: All zeros
            edge_cases.append(np.zeros((3, max(initial_density_shape))))
        elif i % 5 == 1:
            # Case: Maximum force applied at a single random location
            single_max_force_profile = np.zeros((3, max(initial_density_shape)))
            row = np.random.randint(0, single_max_force_profile.shape[0])
            col = np.random.randint(0, single_max_force_profile.shape[1])
            single_max_force_profile[row, col] = force_max
            edge_cases.append(single_max_force_profile)
        elif i % 5 == 2:
            # Case: Random forces with maximum magnitude
            random_force_profile = np.random.uniform(-force_max, force_max, (3, max(initial_density_shape)))
            edge_cases.append(random_force_profile)
        elif i % 5 == 3:
            # Case: Alternating positive and negative forces
            alternating_force_profile = np.zeros((3, max(initial_density_shape)))
            for r in range(alternating_force_profile.shape[0]):
                for c in range(alternating_force_profile.shape[1]):
                    alternating_force_profile[r, c] = (-1)**(r + c) * force_max
            edge_cases.append(alternating_force_profile)
        elif i % 5 == 4:
            # Case: Maximum force applied at multiple random locations
            multi_max_force_profile = np.zeros((3, max(initial_density_shape)))
            num_locations = np.random.randint(1, 5)  # Random number of locations
            for _ in range(num_locations):
                row = np.random.randint(0, multi_max_force_profile.shape[0])
                col = np.random.randint(0, multi_max_force_profile.shape[1])
                multi_max_force_profile[row, col] = force_max
            edge_cases.append(multi_max_force_profile)
    return edge_cases

def _run_edge_case_sample(args: Tuple[int, str, np.ndarray, int, float, Dict, np.ndarray]) -> Dict:
    """
    Run a single edge case sample of the forward model simulation.
    """
    i, output_dir, initial_density, time_steps, dt, parameters, force_profile = args

    e = None  # Initialize e to ensure it is always defined
    try:
        output = forward_model(force_profile, initial_density, time_steps, dt, parameters)
    except Exception as ex:
        e = ex
        logging.error(f"Error in edge case {i}: {e} | Force profile: {force_profile}")
        output = np.full(initial_density.shape, np.nan)
        
    return serialize_data(
        serial_number=i + 1,
        force_profile=force_profile,
        result=output,
        error=str(e) if isinstance(e, Exception) else None
    )

def generate_edge_case_data(
    output_dir: str, 
    num_samples: int = 5
) -> None:
    """
    Generates edge case data by creating specific force profiles, running a forward model,
    and saving the results to a timestamped file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"edge_case_data_{timestamp}.json")

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

    force_profiles = _generate_edge_case_force_profiles(initial_density.shape)
    args = [(i, output_dir, initial_density, time_steps, dt, parameters, force_profiles[i]) for i in range(num_samples)]

    logging.info("Starting edge case simulation...")

    with Pool(processes=cpu_count()) as pool:
        results = []
        for idx, result in enumerate(pool.imap_unordered(_run_edge_case_sample, args), 1):
            results.append(result)
            # Progress bar
            progress = (idx / num_samples) * 100
            logging.info(f"Progress: [{'#' * int(progress // 2)}{'.' * (50 - int(progress // 2))}] {progress:.2f}%")

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"Edge case data saved to {filepath}")

def main() -> None:
    """
    Main function to parse arguments and generate edge case data.
    """
    parser = argparse.ArgumentParser(description="Generate edge case data for the forward model.")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/gijs/Desktop/Thesis/data/raw", 
        help="Directory to save the generated edge case data."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=2500, 
        help="Number of edge cases to generate."
    )
    args = parser.parse_args()

    set_log_level(LogLevel.ERROR)  # Suppress FEniCS log messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    generate_edge_case_data(output_dir=args.output_dir, num_samples=args.num_samples)

if __name__ == "__main__":
    main()
