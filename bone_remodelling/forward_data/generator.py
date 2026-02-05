"""Data generator for bone remodeling simulations."""

import argparse
import logging
import os

from bone_remodelling.parameters import ConfigurationParameters

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, get_context
from pathlib import Path

import numpy as np
from fenics import LogLevel, set_log_level

from bone_remodelling.forward_data.force_profile_generator import (
    ForceProfileGenerator,
)
from bone_remodelling.forward_data.forward_data_manager import (
    ForwardDataManager,
)
from bone_remodelling.forward_model.main import (
    DensitySimulation,
)
from bone_remodelling.forward_model.parameters import (
    SimulationParameters,
)

set_log_level(LogLevel.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_worker_sim: DensitySimulation
_profile_length: int


def init_worker(simulation_parameters: SimulationParameters) -> None:
    """Initialize the per-process simulation (no RNG here)."""
    global _worker_sim, _profile_length  # noqa: PLW0603

    _worker_sim = DensitySimulation(parameters=simulation_parameters)
    _profile_length = simulation_parameters.force_profile.shape[1]


def run_worker_batches(worker_args: list[tuple[int, np.ndarray]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run a batch of simulations in a worker process."""
    force_profiles = []
    final_densities = []

    for i, force_profile in worker_args:
        try:
            _worker_sim.reset()
            _worker_sim.update_force_profile(force_profile)
            _worker_sim.run()
            density = _worker_sim.get_density()
            force_profiles.append(force_profile)
            final_densities.append(density)
        except (RuntimeError, ValueError) as e:  # noqa: PERF203
            logger.error(f"Error processing sample {i + 1}: {e}")
    return force_profiles, final_densities

class TrainingDataGenerator:
    """Class for generating training data for bone remodeling simulations."""

    def __init__(
        self,
        force_profiles: np.ndarray,
        simulation_parameters: SimulationParameters,
        forward_data_manager: ForwardDataManager,
    ) -> None:
        """Initialize the TrainingDataGenerator."""
        self.force_profiles: np.ndarray = force_profiles
        self.num_samples = force_profiles.shape[0]
        self.forward_data_manager = forward_data_manager
        self.simulation_parameters = simulation_parameters

        self._validate_input()

    def _validate_input(self) -> None:
        """Validate the input parameters."""
        if not isinstance(self.force_profiles, np.ndarray):
            raise TypeError("force_profiles must be of type numpy ndarray.")
        if not isinstance(self.simulation_parameters, SimulationParameters):
            raise TypeError(
                "simulation_parameters must be of type SimulationParameters.",
            )

    def generate_parallel(self, max_chunk_size: int = 100) -> None:
        """Generate training data in parallel."""
        cpus_left: int = 2
        num_workers = min(cpu_count() - cpus_left, self.num_samples)
        all_indices = list(range(self.num_samples))

        chunk_size = min(
            (self.num_samples + num_workers - 1) // num_workers,
            max_chunk_size,
        )

        if chunk_size == max_chunk_size:
            logger.warning(
                f"Chunk size capped at max_chunk_size={max_chunk_size}. "
                f"This may result in more tasks than workers.",
            )

        worker_chunks = [
            [
                (i, self.force_profiles[i])
                for i in all_indices[start : start + chunk_size]
            ]
            for start in range(0, self.num_samples, chunk_size)
        ]

        init_args = (self.simulation_parameters,)

        ctx = get_context("fork" if os.name != "nt" else "spawn")
        with ProcessPoolExecutor(
            mp_context=ctx,
            max_workers=num_workers,
            initializer=init_worker,
            initargs=init_args,
        ) as executor:
            futures = [
                executor.submit(run_worker_batches, chunk) for chunk in worker_chunks
            ]

            completed_samples = 0
            for fut in as_completed(futures):
                batch_force_profiles, batch_final_densities = fut.result()
                self.forward_data_manager.save_data(batch_force_profiles, batch_final_densities)
                completed_samples += len(batch_force_profiles)
                pct = completed_samples / self.num_samples * 100
                bar = "#" * int(pct // 2) + "." * (50 - int(pct // 2))
                logger.info(f"Progress: [{bar}] {pct:5.1f}%")

    def generate_serial(self, save_threshold: int = 1000) -> None:
        """Generate training data serially."""
        force_profiles = []
        final_densities = []

        simulation = DensitySimulation(parameters=self.simulation_parameters)
        for i, profile in enumerate(self.force_profiles):
            try:
                simulation.reset()
                simulation.update_force_profile(profile)
                simulation.run()
                density = simulation.get_density()
                force_profiles.append(profile)
                final_densities.append(density)
            except (RuntimeError, ValueError) as e:
                logger.error(f"Error processing sample {i + 1}: {e}")
            # Progress bar
            pct = (i + 1) / self.num_samples * 100
            bar = "#" * int(pct // 2) + "." * (50 - int(pct // 2))
            logger.info(f"Progress: [{bar}] {pct:5.1f}%")
            if len(force_profiles) >= save_threshold:
                self.forward_data_manager.save_data(force_profiles, final_densities)
                force_profiles = []
                final_densities = []
        if force_profiles:
            self.forward_data_manager.save_data(force_profiles, final_densities)

def cli(config: ConfigurationParameters, argv: list[str]) -> None:
    """CLI entry point for training data generation."""
    parser = argparse.ArgumentParser(description="Generate training data for bone remodeling simulations.")
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--force_type",
        type=str,
        choices=["merger", "triangular"],
        default="triangular",
        help="Type of force profile to generate ('merger' or 'triangular').",
    )
    args = parser.parse_args(argv)
    run(config, samples=args.samples, force_type=args.force_type)

def run(config: ConfigurationParameters, samples: int, force_type: str) -> None:
    """Generate training data for bone remodeling simulations.

    Args:
    ----
        config (ConfigurationParameters): Configuration parameters for the simulation.
        samples (int): Number of samples to generate.
        force_type (str): Type of force profile to generate ('merger' or 'triangular').

    """
    initial_density = np.full((config.mesh_top_resolution, config.mesh_side_resolution), config.start_density)
    logger.info("Generating force profiles...")

    force_profile_generator = ForceProfileGenerator(
        profile_top_and_sides=(config.force_top_resolution, config.force_side_resolution),
        batch_seed=config.seed,
    )
    force_mask = force_profile_generator.generate_force_mask()

    force_profiles, directory = None, config.output_dir / Path("raw")
    if force_type == "merger":
        force_profiles = force_profile_generator.merger(
            num_samples=samples,
        )
    elif force_type == "triangular":
        force_profiles = force_profile_generator.triangular_only(
            num_samples=samples,
        )
        directory = directory / Path("triangular")
    forward_data_manager = ForwardDataManager(storage_path=directory)

    if force_mask is None or force_profiles is None:
        logger.error("Failed to generate force profiles or force mask.")
        return

    logger.info("Running forward model simulations...")

    empty_force_profile = np.zeros((3, np.max(initial_density.shape)))
    simulation_parameters = SimulationParameters(
        force_profile=empty_force_profile,
        force_mask=force_mask,
        initial_density_field=initial_density,
        min_density=config.min_density,
        max_density=config.max_density,
    )

    data_generator = TrainingDataGenerator(
        force_profiles=force_profiles,
        simulation_parameters=simulation_parameters,
        forward_data_manager=forward_data_manager,
    )
    start_time = time.time()
    try:
        data_generator.generate_parallel(max_chunk_size=500)
    except Exception:  # noqa: BLE001
        logger.warning("An error occurred during parallel data generation", exc_info = True)
        logger.warning("Continuing with serialised data generation instead (note: significantly slower)")
        data_generator.generate_serial()
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logger.info(f"Simulation completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    # Developer convenience entry point.
    # For reproducible runs, use the unified CLI (main.py).
    config = ConfigurationParameters(
        output_dir=Path(__file__).resolve().parent.parent.parent / Path("data"),
    )
    samples = 1_000
    force_type = "triangular"
    run(config, samples, force_type)
