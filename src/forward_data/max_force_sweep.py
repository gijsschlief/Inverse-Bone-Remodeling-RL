"""Compute Shannon energy entropy of density profiles. Which is used to determine max force based on a reduction in entropy by oversaturation of the maximum density threshold."""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from bone_remodeling.src.forward_data.force_profile_generator import (
    ForceProfileGenerator,  # type: ignore
)
from bone_remodeling.src.forward_data.generator import (
    TrainingDataGenerator,  # type: ignore
)
from bone_remodeling.src.forward_data.reader import (
    forward_data_reader,  # type: ignore
)
from bone_remodeling.src.forward_model.parameters import (
    SimulationParameters,  # type: ignore
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def shannon_energy_entropy(density: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of the energy-normalised density field."""
    e = density.ravel() ** 2
    p = e / (e.sum() + eps)
    p = p[p > eps]      # remove zeros
    return -np.sum(p * np.log(p))

def generate_force_batches(force_max_values: list[float], batch_size: int = 20) -> list[np.ndarray]:
    """Generate batches of force profiles for given max force values."""
    gen = ForceProfileGenerator(profile_length=10, batch_seed=42)
    batches = []

    for fmax in force_max_values:
        profiles = gen.merger(num_samples=batch_size, force_max=fmax)
        batches.append(profiles)

    return batches

def run_sweep() -> None:
    """Run the max force sweep simulations."""
    initial_density = np.full((10, 10), 0.8)

    logger.info("Generating force profiles...")

    # Generate a set of force profiles for the sweep that increase in maximum force
    force_max_values = np.linspace(0.0, 100.0, num=100)
    force_profile_batches = generate_force_batches(force_max_values, batch_size=20)
    force_profiles = np.vstack(force_profile_batches)
    logger.info("Running forward model simulations...")

    empty_force_profile = np.zeros((3, np.max(initial_density.shape)))
    simulation_parameters = SimulationParameters(
        force_profile=empty_force_profile,
        initial_density_field=initial_density,
    )

    data_generator = TrainingDataGenerator(
        force_profiles=force_profiles,
        output_dir="/home/gijs/Desktop/Thesis/data/raw",
        simulation_parameters=simulation_parameters,
    )
    start_time = time.time()
    _ = data_generator.generate_parallel(
        max_chunk_size=500,
        force_profile_name="force_sweep",
    )
    stop_time = time.time()

    elapsed_time = stop_time - start_time
    logger.info(f"Simulation completed in {elapsed_time:.2f} seconds.")

def analyse_sweep() -> None:
    """Analyse the results of the max force sweep simulations."""
    data = forward_data_reader(
        file_path="/home/gijs/Desktop/Thesis/data/raw/training_force_sweep_10_samples_1203_1403.json",
    )

    if data is None:
        logger.error("No data found for analysis.")
        return
    _, force_profiles, densities = data

    entropies = []
    max_forces = []

    for force_profile, final_density in zip(force_profiles, densities):
        max_force = np.max(force_profile)
        entropy = shannon_energy_entropy(final_density)

        max_forces.append(max_force)
        entropies.append(entropy)

    plot_results(max_forces, entropies)

def plot_results(max_forces: list[float], entropies: list[float]) -> None:
    """Plot the max forces against the entropies."""
    plt.figure(figsize=(8, 6))
    plt.scatter(max_forces, entropies, marker='o')
    plt.xlabel('Maximum Force')
    plt.ylabel('Shannon Energy Entropy')
    plt.title('Max Force vs Shannon Energy Entropy')
    plt.grid(visible=True)
    plt.show()

if __name__ == "__main__":
    run_sweep()
    analyse_sweep()
