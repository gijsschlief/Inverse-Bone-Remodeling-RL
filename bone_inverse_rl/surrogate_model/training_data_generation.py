import numpy as np
import datetime
import os
import json
from bone_inverse_rl.forward_model.forward_model import forward_model
from typing import Dict, Optional, Tuple
from fenics import set_log_level, LogLevel

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
        force_profile_length (int): Length of each force profile.

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
        output = forward_model(force_profile, initial_density, time_steps, dt, parameters)
        data_point = serialize_data(
            force_profile=force_profile, 
            result=output, 
            serial_number=np.array([i + 1])
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
            print(f"\nAverage time per sample (1-11): {avg_time_per_sample:.4f} seconds")

        # Display progress bar
        progress = (i / num_samples) * 100
        print(f"\rProgress: [{'#' * int(progress // 2)}{'.' * (50 - int(progress // 2))}] {progress:.2f}%", end="")

    print(f"Training data saved to {filepath}")
    return None

def serialize_data(
    force_profile: np.ndarray, 
    result: np.ndarray, 
    serial_number: np.ndarray) -> Dict[str, np.ndarray]:
    return {
    "serial_number": serial_number.tolist(),
    "force_profile": force_profile.tolist(),
    "final_output_density": result.tolist()
    }

# Example usage
if __name__ == "__main__":
    set_log_level(LogLevel.ERROR)  # Suppress FEniCS log messages
    generate_training_data(output_dir="/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/data/raw", num_samples=100)