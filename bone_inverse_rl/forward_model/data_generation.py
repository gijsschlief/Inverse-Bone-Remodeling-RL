import argparse
import datetime
import os
import json
import logging
from typing import Dict

import numpy as np
from fenics import set_log_level, LogLevel

from Thesis_code.bone_inverse_rl.forward_model.main import forward_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_training_data(
    output_dir: str, 
    num_samples: int = 100
) -> None:

    """
    Generates training data by creating random force profiles, running a forward model,
    and saving the results to a timestamped file.

    Args:
        output_dir (str): Directory to save the output file.
        num_samples (int): Number of random force profiles to generate.

    Returns:
        None: The function saves the training data to a JSON file in the specified directory.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_data_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Define parameters for the forward model
    initial_density = np.full((10, 10), 0.8) 
    time_steps = 100
    dt = 1.0 
    parameters = {
        'file_location': output_dir,
        'rho_min': 0.01,
        'rho_max': 1.74,
    }

    # Generate random force profiles and collect results
    data = []
    for i in range(num_samples):
        force_profile = np.zeros((3, max(initial_density.shape)))
        force_max = 3  # Define the maximum force value
        num_forces = np.random.randint(1, 4)  # Random number of forces between 1 and 3

        # Randomly select three unique locations on the side of the density matrix
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

def serialize_data(
    serial_number: int, 
    force_profile: np.ndarray, 
    result: np.ndarray, 
    error: str = None) -> Dict:
    """
    Serialize the data into a dictionary format for saving or further processing.
    If there is no error, exclude the error field from the dictionary.
    """
    serialized_data = {
        "serial_number": serial_number,
        "force_profile": force_profile.tolist(),
        "final_output_density": result.tolist(),
    }
    if error is not None:
        serialized_data["error"] = error
    return serialized_data

def main():
    parser = argparse.ArgumentParser(description="Generate training data for the forward model.")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/gijs/Desktop/Thesis/data/raw", 
        help="Directory to save the output file."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100, 
        help="Number of random force profiles to generate (default: 100)."
    )
    args = parser.parse_args()

    set_log_level(LogLevel.ERROR)  # Suppress FEniCS log messages
    generate_training_data(output_dir=args.output_dir, num_samples=args.num_samples)
    return

if __name__ == "__main__":
    main()