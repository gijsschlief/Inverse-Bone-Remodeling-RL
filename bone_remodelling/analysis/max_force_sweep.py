"""Compute Shannon energy entropy of density profiles. Which is used to determine max force based on a reduction in entropy by oversaturation of the maximum density threshold."""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from bone_remodelling.forward_data.force_profile_generator import (
    ForceProfileGenerator,
)
from bone_remodelling.forward_data.generator import (
    TrainingDataGenerator,
)
from bone_remodelling.forward_data.reader import (
    forward_data_reader,
)
from bone_remodelling.forward_model.parameters import (
    SimulationParameters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def shannon_energy_entropy(density: np.ndarray) -> float:
    """Shannon entropy of the energy-normalised density field."""
    e = density.ravel() ** 2
    p = e / e.sum()
    return -np.sum(p * np.log(p))

def matrix_saturation(density: np.ndarray, max_density: float = 1.74) -> np.bool:
    """Calculate the saturation level of the density matrix."""
    return np.sum(density >= max_density)

def matrix_undersaturation(density: np.ndarray, min_density: float = 0.01) -> np.bool:
    """Calculate the level of undersaturation of the density matrix."""
    return np.sum(density <= min_density)

def generate_force_batches(force_max_values: list[float], batch_size: int = 20) -> list[np.ndarray]:
    """Generate batches of force profiles for given max force values."""
    gen = ForceProfileGenerator(profile_top=10, batch_seed=42)
    batches = []

    for _ in force_max_values:
        profiles = gen.merger(num_samples=batch_size)
        batches.append(profiles)

    return batches

def run_sweep() -> None:
    """Run the max force sweep simulations."""
    initial_density = np.full((10, 10), 0.8)

    logger.info("Generating force profiles...")
    generator = ForceProfileGenerator(profile_top=10, batch_seed=42)
    force_profiles = generator.merger(num_samples=500)

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

def plot_oversaturation(x: np.ndarray, y: np.ndarray, bin_centers: list[float], bin_means: list[float], saturation_threshold: float = 5e4) -> None:
    """Plot oversaturation analysis from the sweep results."""
    plt.figure(num=1,figsize=(8, 6))
    plt.scatter(x, y, s=10, alpha=0.4, label="Raw samples")
    plt.plot(bin_centers, bin_means, "-o", color="red", label="Binned trend")
    plt.axvline(x=saturation_threshold, color="black", linestyle="--", label=f"Saturation Threshold {saturation_threshold:.1E}")
    plt.xlabel("Force Profile Energy")
    plt.ylabel("Matrix Oversaturation")
    plt.xscale("log")
    plt.xlim(left=1)
    plt.title("Energy vs Matrix Oversaturation")
    plt.grid(visible=True)
    plt.legend()

def plot_undersaturation(x: np.ndarray, y: np.ndarray, bin_centers: list[float], bin_means: list[float], saturation_threshold: float = 1e2) -> None:
    """Plot undersaturation analysis from the sweep results."""
    plt.figure(num=2,figsize=(8, 6))
    plt.scatter(x, y, s=10, alpha=0.4, label="Raw samples")
    plt.plot(bin_centers, bin_means, "-o", color="red", label="Binned trend")
    plt.axvline(x=saturation_threshold, color="black", linestyle="--", label=f"Saturation Threshold {saturation_threshold:.1E}")
    plt.xlabel("Force Profile Energy")
    plt.ylabel("Matrix Undersaturation")
    plt.xscale("log")
    plt.xlim(left=1)
    plt.title("Energy vs Matrix Undersaturation")
    plt.grid(visible=True)
    plt.legend()
    plt.show()

def analyse_sweep() -> None:
    """Analyse the results of the max force sweep simulations."""
    data = forward_data_reader(
        file_path="/home/gijs/Desktop/Thesis/data/sweeps/",
    )

    if data is None:
        logger.error("No data found for analysis.")
        return
    _, force_profiles, densities = data

    saturations: list[np.bool] = []
    undersaturations: list[np.bool] = []
    energies: list[float] = []

    for force_profile, final_density in zip(force_profiles, densities):
        energy = force_profile_energy(force_profile)
        saturation = matrix_saturation(final_density)
        undersaturation = matrix_undersaturation(final_density)
        energies.append(energy)
        saturations.append(saturation)
        undersaturations.append(undersaturation)

    energies_array = np.array(energies)
    saturations_array = np.array(saturations)
    undersaturations_array = np.array(undersaturations)

    # Sort by energy
    idx = np.argsort(energies_array)
    x = energies_array[idx]
    y_oversaturated = saturations_array[idx]
    total_bins: int = 20
    bins = np.logspace(np.log10(1), np.log10(10e8), total_bins + 1)

    bin_centers_oversaturated = []
    bin_means_oversaturated = []

    for i in range(total_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.any(mask):
            bin_centers_oversaturated.append((bins[i] + bins[i + 1]) / 2)
            bin_means_oversaturated.append(np.mean(y_oversaturated[mask]))

    plot_oversaturation(x, y_oversaturated, bin_centers_oversaturated, bin_means_oversaturated)

    # THE SAME BUT FOR UNDERSATURATION
    y_undersaturated = undersaturations_array[idx]
    bins = np.logspace(np.log10(1), np.log10(10e8), total_bins + 1)

    bin_centers_undersaturated = []
    bin_means_undersaturated = []

    for i in range(total_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.any(mask):
            bin_centers_undersaturated.append((bins[i] + bins[i + 1]) / 2)
            bin_means_undersaturated.append(np.mean(y_undersaturated[mask]))

    plot_undersaturation(x, y_undersaturated, bin_centers_undersaturated, bin_means_undersaturated)

if __name__ == "__main__":
    run_sweep()
    analyse_sweep()
