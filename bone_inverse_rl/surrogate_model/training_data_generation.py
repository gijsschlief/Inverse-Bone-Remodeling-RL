import numpy as np
import datetime
import os
import json
from bone_inverse_rl.forward_model.forward_model_train import forward_model_train

def generate_training_data(output_dir, num_samples=100, force_profile_length=50):
    """
    Generates training data by creating random force profiles, running a forward model,
    and saving the results to a timestamped file.

    Args:
        output_dir (str): Directory to save the output file.
        num_samples (int): Number of random force profiles to generate.
        force_profile_length (int): Length of each force profile.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

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
    for _ in range(num_samples):
        force_profile = np.zeros((3, max(initial_density.shape)))
        force_max = 10  # Define the maximum force value
        num_forces = np.random.randint(1, 4)  # Random number of forces between 1 and 3

        # Randomly select three unique locations in the matrix
        locations = np.random.choice(np.prod(initial_density.shape), num_forces, replace=False)
        for loc in locations:
            row, col = divmod(loc, initial_density.shape[1])
            force_profile[row, col] = np.random.uniform(-force_max, force_max)
        result = forward_model_train(force_profile, initial_density, time_steps, dt, parameters)
        data.append({"force_profile": force_profile.tolist(), "result": result})

    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_data_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Save the data to a JSON file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Training data saved to {filepath}")

# Example usage
if __name__ == "__main__":
    generate_training_data(output_dir="/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/data/raw", num_samples=100, force_profile_length=50)