"""Data generator for bone remodeling simulations."""

import datetime
import json
import logging
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from bone_remodeling.src.forward_data.force_profile_generator import (
    ForceProfileGenerator,  # type: ignore
)
from bone_remodeling.src.forward_model.main import (
    DensitySimulation,  # type: ignore
)
from bone_remodeling.src.forward_model.parameters import (
    SimulationParameters,  # type: ignore
)
from fenics import LogLevel, set_log_level  # type: ignore

set_log_level(LogLevel.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_worker_sim: DensitySimulation
_profile_length: int


def init_worker(simulation_parameters: SimulationParameters) -> None:
    """Initialize the per-process simulation (no RNG here)."""
    global _worker_sim, _profile_length

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
                }
            )
        except Exception as e:
            logging.error(f"Error processing sample {i + 1}: {e}")
            results.append(
                {
                    "serial_number": i + 1,
                    "force_profile": force_profile.tolist(),
                    "final_output_density": None,
                    "error": str(e),
                }
            )
    return results


class TrainingDataGenerator:
    """Class for generating training data for bone remodeling simulations."""

    def __init__(
        self,
        force_profiles: np.ndarray,
        output_dir: str,
        simulation_parameters: SimulationParameters,
    ) -> None:
        """Initialize the TrainingDataGenerator."""
        self.force_profiles: np.ndarray = force_profiles
        self.num_samples = force_profiles.shape[0]
        self.output_dir: Path = Path(output_dir)
        self.simulation_parameters = simulation_parameters

        self._validate_input()
        os.makedirs(self.output_dir, exist_ok=True)

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
                "simulation_parmaeters must be of type SimulationParameters."
            )

    def generate_parallel(
        self, max_chunk_size: int = 100, force_profile_name: str = "Undefined"
    ) -> list[dict]:
        """Generate training data in parallel."""
        num_workers = min(cpu_count(), self.num_samples)
        all_indices = list(range(self.num_samples))

        chunk_size = min(
            (self.num_samples + num_workers - 1) // num_workers, max_chunk_size
        )

        if chunk_size == max_chunk_size:
            logging.warning(
                f"Chunk size capped at max_chunk_size={max_chunk_size}. "
                f"This may result in more tasks than workers."
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

        ctx = get_context("fork")
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
                logging.info(f"Progress: [{bar}] {pct:5.1f}%")

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
                    }
                )
            except Exception as e:
                logging.error(f"Error processing sample {i + 1}: {e}")
                results.append(
                    {
                        "serial_number": i + 1,
                        "force_profile": profile.tolist(),
                        "final_output_density": None,
                        "error": str(e),
                    }
                )
            # Progress bar
            pct = (i + 1) / self.num_samples * 100
            bar = "#" * int(pct // 2) + "." * (50 - int(pct // 2))
            logging.info(f"Progress: [{bar}] {pct:5.1f}%")

        self._save_results(results, force_profile_name)
        return results

    def _save_results(self, results: list[dict], force_profile_name: str) -> None:
        """Save the results to a JSON file."""
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            "%m%d_%H%M"
        )
        filepath = (
            self.output_dir
            / f"training_{force_profile_name}_{self.num_samples}_samples_{timestamp}.json"
        )

        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Training data saved to {filepath}")

    @staticmethod
    def serialize_data(
        serial_number: int,
        force_profile: np.ndarray,
        result: np.ndarray,
        error: Optional[str] = None,
    ) -> Dict:
        """Serialize the data for a single sample."""
        serialized_data = {
            "serial_number": serial_number,
            "force_profile": force_profile.tolist(),
            "final_output_density": result.tolist(),
        }
        if error is not None:
            serialized_data["error"] = error
        return serialized_data


if __name__ == "__main__":
    import time

    initial_density = np.full((10, 10), 0.8)

    logging.info("Generating force profiles...")

    force_profile_generator = ForceProfileGenerator(
        profile_length=10, batch_seed=np.random.randint(0, 1_000_000)
    )

    force_profiles = force_profile_generator.merger(
        num_samples=30_000,
        force_max=20.0,
    )

    logging.info("Running forward model simulations...")

    empty_force_profile = np.zeros((3, np.max(initial_density.shape)))
    simulation_parameters = SimulationParameters(
        force_profile=empty_force_profile, initial_density_field=initial_density
    )

    data_generator = TrainingDataGenerator(
        force_profiles=force_profiles,
        output_dir="/home/gijs/Desktop/Thesis/data/raw",
        simulation_parameters=simulation_parameters,
    )
    start_time = time.time()
    _ = data_generator.generate_parallel(
        max_chunk_size=500, force_profile_name="combined_third_order"
    )
    # _ = data_generator.generate_serial()
    stop_time = time.time()

    elapsed_time = stop_time - start_time
    logging.info(f"Simulation completed in {elapsed_time:.2f} seconds.")
