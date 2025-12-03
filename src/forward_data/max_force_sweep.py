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

def shannon_energy_entropy(density: np.ndarray) -> float:
    """Shannon entropy of the energy-normalised density field."""
    e = density.ravel() ** 2
    p = e / e.sum()
    return -np.sum(p * np.log(p))

def matrix_saturation(density: np.ndarray, max_density: float = 1.74) -> int:
    """Calculate the saturation level of the density matrix."""
    return np.sum(density >= max_density)

def generate_force_batches(force_max_values: list[float], batch_size: int = 20) -> list[np.ndarray]:
    """Generate batches of force profiles for given max force values."""
    gen = ForceProfileGenerator(profile_length=10, batch_seed=42)
    batches = []

    for fmax in force_max_values:
        profiles = gen.merger(num_samples=batch_size, scaling=fmax)
        batches.append(profiles)

    return batches

def run_sweep() -> None:
    """Run the max force sweep simulations."""
    initial_density = np.full((10, 10), 0.8)

    logger.info("Generating force profiles...")

    # Generate a set of force profiles for the sweep that increase in maximum force
    force_max_values = np.linspace(200, 500.0, num=20)
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
        output_dir="/home/gijs/Desktop/Thesis/data/sweeps",
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

def force_profile_energy(force_profile: np.ndarray) -> float:
    """Calculate the energy of a force profile using the sum of all values squared."""
    return np.sum(force_profile ** 2)

def analyse_sweep() -> None:
    """Analyse the results of the max force sweep simulations."""
    data = forward_data_reader(
        file_path="/home/gijs/Desktop/Thesis/data/sweeps/",
    )

    if data is None:
        logger.error("No data found for analysis.")
        return
    _, force_profiles, densities = data

    saturations = []
    energies = []

    for force_profile, final_density in zip(force_profiles, densities):
        energy = force_profile_energy(force_profile)
        saturation = matrix_saturation(final_density)
        energies.append(energy)
        saturations.append(saturation)

    energies = np.array(energies)
    saturations = np.array(saturations)

    # Sort by energy
    idx = np.argsort(energies)
    x = energies[idx]
    y = saturations[idx]
    N_bins: int = 20
    bins = np.logspace(np.log10(1), np.log10(10e8), N_bins + 1)

    bin_centers = []
    bin_means = []

    for i in range(N_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.any(mask):
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_means.append(np.mean(y[mask]))

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=10, alpha=0.4, label="Raw samples")
    plt.plot(bin_centers, bin_means, "-o", color="red", label="Binned trend")

    # Add vertical line at saturation threshold
    saturation_threshold = 5e4  # Example threshold value
    plt.axvline(x=saturation_threshold, color="black", linestyle="--", label=f"Saturation Threshold {saturation_threshold:.1E}")

    plt.xlabel("Force Profile Energy")
    plt.ylabel("Matrix Saturation")
    plt.xscale("log")
    plt.xlim(left=1)
    plt.title("Energy vs Matrix Saturation")
    plt.grid(visible=True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #run_sweep()
    analyse_sweep()
