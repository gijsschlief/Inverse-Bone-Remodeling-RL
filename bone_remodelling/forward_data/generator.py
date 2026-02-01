"""Data generator for bone remodeling simulations."""

import datetime
import json
import logging
import os

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


def run_worker_batches(worker_args: list[tuple[int, np.ndarray]]) -> list[dict]:
    """Run a batch of simulations in a worker process."""
    results = []
    for i, force_profile in worker_args:
        try:
            _worker_sim.reset()
            _worker_sim.update_force_profile(force_profile)
            _worker_sim.run()
            density = _worker_sim.get_density()
            results.append(
                {
                    "serial_number": i + 1,
                    "force_profile": force_profile.tolist(),
                    "final_output_density": density.tolist(),
                },
            )
        except (RuntimeError, ValueError) as e:  # noqa: PERF203
            logger.error(f"Error processing sample {i + 1}: {e}")
            results.append(
                {
                    "serial_number": i + 1,
                    "force_profile": force_profile.tolist(),
                    "final_output_density": None,
                    "error": str(e),
                },
            )
    return results


class TrainingDataGenerator:
    """Class for generating training data for bone remodeling simulations."""

    def __init__(
        self,
        force_profiles: np.ndarray,
        simulation_parameters: SimulationParameters,
        output_dir: Path = Path(__file__).parent.parent.parent / Path("data", "raw"),
    ) -> None:
        """Initialize the TrainingDataGenerator."""
        self.force_profiles: np.ndarray = force_profiles
        self.num_samples = force_profiles.shape[0]
        self.output_dir: Path = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.simulation_parameters = simulation_parameters

        self._validate_input()

    def _validate_input(self) -> None:
        """Validate the input parameters."""
        if not isinstance(self.force_profiles, np.ndarray):
            raise TypeError("force_profiles must be of type numpy ndarray.")
        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be of type Path")
        if not self.output_dir.exists():
            raise ValueError(f"Output directory {self.output_dir} does not exist.")
        if not isinstance(self.simulation_parameters, SimulationParameters):
            raise TypeError(
                "simulation_parameters must be of type SimulationParameters.",
            )

    def generate_parallel(
        self,
        max_chunk_size: int = 100,
        force_profile_name: str = "Undefined",
    ) -> list[dict]:
        """Generate training data in parallel."""
        num_workers = min(cpu_count(), self.num_samples)
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
        results = []

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
                batch_results = fut.result()
                results.extend(batch_results)
                completed_samples += len(batch_results)
                pct = completed_samples / self.num_samples * 100
                bar = "#" * int(pct // 2) + "." * (50 - int(pct // 2))
                logger.info(f"Progress: [{bar}] {pct:5.1f}%")

        self._save_results(results, force_profile_name)
        return results

    def generate_serial(self, force_profile_name: str = "Undefined") -> list[dict]:
        """Generate training data serially."""
        results = []

        simulation = DensitySimulation(parameters=self.simulation_parameters)
        for i, profile in enumerate(self.force_profiles):
            try:
                simulation.reset()
                simulation.update_force_profile(profile)
                simulation.run()
                density = simulation.get_density()
                results.append(
                    {
                        "serial_number": i + 1,
                        "force_profile": profile.tolist(),
                        "final_output_density": density.tolist(),
                    },
                )
            except (RuntimeError, ValueError) as e:
                logger.error(f"Error processing sample {i + 1}: {e}")
                results.append(
                    {
                        "serial_number": i + 1,
                        "force_profile": profile.tolist(),
                        "final_output_density": None,
                        "error": str(e),
                    },
                )
            # Progress bar
            pct = (i + 1) / self.num_samples * 100
            bar = "#" * int(pct // 2) + "." * (50 - int(pct // 2))
            logger.info(f"Progress: [{bar}] {pct:5.1f}%")

        self._save_results(results, force_profile_name)
        return results

    def _save_results(self, results: list[dict], force_profile_name: str) -> None:
        """Save the results to a JSON file."""
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            "%m%d_%H%M",
        )
        filepath = (
            self.output_dir
            / f"training_{force_profile_name}_{self.num_samples}_samples_{timestamp}.json"
        )

        with Path.open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Training data saved to {filepath}")

    @staticmethod
    def serialize_data(
        serial_number: int,
        force_profile: np.ndarray,
        result: np.ndarray,
        error: str | None = None,
    ) -> dict:
        """Serialize the data for a single sample."""
        serialized_data = {
            "serial_number": serial_number,
            "force_profile": force_profile.tolist(),
            "final_output_density": result.tolist(),
        }
        if error is not None:
            serialized_data["error"] = error
        return serialized_data

def main(samples: int, force_type: str) -> None:
    """Generate training data for bone remodeling simulations.

    Args:
    ----
        samples (int): Number of samples to generate.
        force_type (str): Type of force profile to generate ('merger' or 'triangular').

    """
    initial_density = np.full((10, 10), 0.87)
    logger.info("Generating force profiles...")

    force_profile_generator = ForceProfileGenerator(
        profile_top_and_sides=(10, 10),
        batch_seed=1,
    )
    force_mask = force_profile_generator.generate_force_mask()

    force_profiles, directory = None, Path(__file__).parent.parent.parent / Path("data", "raw")
    if force_type == "merger":
        force_profiles = force_profile_generator.merger(
            num_samples=samples,
        )
    elif force_type == "triangular":
        force_profiles = force_profile_generator.triangular_only(
            num_samples=samples,
        )
        directory = directory / Path("triangular")
    if force_mask is None or force_profiles is None:
        logger.error("Failed to generate force profiles or force mask.")
        return

    logger.info("Running forward model simulations...")

    empty_force_profile = np.zeros((3, np.max(initial_density.shape)))
    simulation_parameters = SimulationParameters(
        force_profile=empty_force_profile,
        force_mask=force_mask,
        initial_density_field=initial_density,
    )

    data_generator = TrainingDataGenerator(
        force_profiles=force_profiles,
        simulation_parameters=simulation_parameters,
        output_dir=directory,
    )
    start_time = time.time()
    try:
        _ = data_generator.generate_parallel(
            max_chunk_size=500,
            force_profile_name="final_run",
        )
    except Exception:  # noqa: BLE001
        logger.warning("An error occurred during parallel data generation", exc_info = True)
        logger.warning("Continuing with serialised data generation instead (note: significantly slower)")
        _ = data_generator.generate_serial()
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logger.info(f"Simulation completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    samples = 1_000
    force_type = "triangular"
    main(samples, force_type)
