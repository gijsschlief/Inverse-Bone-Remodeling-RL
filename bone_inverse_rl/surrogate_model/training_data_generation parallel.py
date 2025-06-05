import numpy as np
import datetime
import os
import json
from bone_inverse_rl.forward_model.forward_model import forward_model
from typing import Dict, Optional, Tuple
from fenics import set_log_level, LogLevel
from multiprocessing import Pool, cpu_count

def _run_one_sample(args) -> Dict:
    i, output_dir, initial_density, time_steps, dt, parameters = args

    force_profile = np.zeros((3, max(initial_density.shape)))
    force_max = 3
    num_forces = np.random.randint(1, 7)
    locations = np.random.choice(np.prod(force_profile.shape), num_forces, replace=False)
    for loc in locations:
        row, col = divmod(loc, force_profile.shape[1])
        force_profile[row, col] = np.random.uniform(-force_max, force_max)

    output = forward_model(force_profile, initial_density, time_steps, dt, parameters)

    return {
        "serial_number": [i + 1],
        "force_profile": force_profile.tolist(),
        "final_output_density": output.tolist()
    }

def generate_training_data(
    output_dir: str, 
    num_samples: int = 100
) -> None:
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
        'save': False  # avoid unnecessary file I/O
    }

    args = [(i, output_dir, initial_density, time_steps, dt, parameters) for i in range(num_samples)]

    print("Starting parallel simulation...")
    set_log_level(LogLevel.ERROR)

    with Pool(processes=cpu_count()) as pool:
        results = []
        for idx, result in enumerate(pool.imap_unordered(_run_one_sample, args), 1):
            results.append(result)
            # Progress bar
            progress = (idx / num_samples) * 100
            print(f"\rProgress: [{'#' * int(progress // 2)}{'.' * (50 - int(progress // 2))}] {progress:.2f}%", end="")

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nTraining data saved to {filepath}")

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
    generate_training_data(output_dir="/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/data/raw", num_samples=25000)