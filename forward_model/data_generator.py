""""Data generator for bone remodeling simulations."""

import datetime
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from bone_remodeling.forward_model.density_simulation import (
    DensitySimulation,  # type: ignore
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_worker_sim: DensitySimulation
_profile_length: int


def init_worker(
    empty_profile: np.ndarray,
    initial_density: np.ndarray,
    time_steps: int,
    dt: float,
    parameters: dict,
) -> None:
    """Initialize the per-process simulation (no RNG here)."""
    global _worker_sim, _profile_length

    _worker_sim = DensitySimulation(
        force_profile=empty_profile,
        initial_density_field=initial_density,
        time_steps=time_steps,
        dt=dt,
        parameters=parameters,
    )

    _profile_length = empty_profile.shape[1]

def run_worker_batches(worker_args: list[tuple[int, np.ndarray]]) -> list[dict]:
    """Run a batch of simulations in a worker process."""
    results = []
    for i, profile in worker_args:
        _worker_sim.reset()
        _worker_sim.update_force_profile(profile)
        _worker_sim.run()
        dens = _worker_sim.get_density()
        results.append({
            "serial_number": i + 1,
            "force_profile": profile.tolist(),
            "final_output_density": dens.tolist(),
        })
    return results

class TrainingDataGenerator:
    """Class for generating training data for bone remodeling simulations."""

    def __init__(
        self,
        output_dir: str,
        initial_density: Union[np.ndarray, None] = None,
        force_max: int = 2,
        force_count_max: int = 7,
        batch_seed: int | None = None,
    ) -> None:
        """Initialize the TrainingDataGenerator."""
        os.environ["OMP_NUM_THREADS"]   = "1"
        os.environ["MKL_NUM_THREADS"]   = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        from fenics import LogLevel, set_log_level
        set_log_level(LogLevel.ERROR)

        self.output_dir: Path = Path(output_dir)
        self.force_max: int = force_max
        self.force_count_max: int = force_count_max
        self.batch_seed: int | None = batch_seed
        self.time_steps: int = 100
        self.dt: float = 1.0
        self.initial_density: np.ndarray = (
            np.full((10, 10), 0.8) if initial_density is None else initial_density
        )
        self.parameters: Dict[str, Any] | None = self._load_parameters()
        self._validate_input()
        os.makedirs(self.output_dir, exist_ok=True)
        self.empty_force_profile = np.zeros((3, np.max(self.initial_density.shape)))
        self._rng = np.random.default_rng()
        self._profile_length = np.max(self.initial_density.shape)

    def _load_parameters(self) -> Dict:
        """Load simulation parameters from a JSON file."""
        parameters_file = Path(__file__).resolve().parent / "parameters.json"
        if parameters_file.exists():
            with open(parameters_file) as f:
                loaded_parameters = json.load(f)
                self.time_steps = loaded_parameters.get("time_steps", self.time_steps)
                self.dt = loaded_parameters.get("dt", self.dt)
                return loaded_parameters
        else:
            logging.warning(
                f"parameters.json not found at {parameters_file}. Using default values.",
            )
            return {}

    def _validate_input(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise ValueError("output_dir must be a Path object.")
        if not self.output_dir.exists():
            raise ValueError(f"Output directory {self.output_dir} does not exist.")
        if not isinstance(self.force_max, int) or self.force_max <= 0:
            raise ValueError("force_max must be a positive integer.")
        if not isinstance(self.initial_density, np.ndarray):
            raise ValueError("initial_density must be a numpy ndarray.")
        if not isinstance(self.force_count_max, int) or self.force_count_max <= 0:
            raise ValueError("force_count_max must be a positive integer.")

    def _generate_random_force_profiles(self, num_samples: int) -> np.ndarray:
        """Generate all random force profiles in a fully vectorized way."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        total_elements = profiles.shape[1] * profiles.shape[2]

        rng = self._rng  # Use a single RNG (already seeded from batch_seed)

        # How many non-zero forces per sample?
        counts = rng.integers(1, self.force_count_max, size=num_samples)
        total_forces = np.sum(counts)

        # Flat indices for force assignment
        all_indices = rng.choice(
            total_elements,
            size=total_forces,
            replace=True  # reuse allowed across different samples
        )

        # Random force values
        all_forces = rng.uniform(-self.force_max, self.force_max, size=total_forces)

        # Assign values back to profiles
        flat_profiles = profiles.reshape(num_samples, -1)
        pointer = 0
        for i, count in enumerate(counts):
            if count > 0:
                flat_profiles[i, all_indices[pointer:pointer+count]] = all_forces[pointer:pointer+count]
                pointer += count

        return profiles

    def generate_parallel(self, num_samples: int) -> None:
        """Generate training data in parallel."""
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%m%d_%H%M")
        filepath = (
            self.output_dir
            / f"training_batch_{self.batch_seed}_samples_{num_samples}_{timestamp}.json"
        )

        force_profiles = self._generate_random_force_profiles(num_samples)

        num_workers = min(cpu_count(), num_samples)
        max_chunk_size = 10
        all_indices = list(range(num_samples))

        # Always respect max_chunk_size
        chunk_size = min(
            (num_samples + num_workers - 1) // num_workers,
            max_chunk_size
        )

        if chunk_size == max_chunk_size:
            logging.warning(
                f"Chunk size capped at max_chunk_size={max_chunk_size}. "
                f"This may result in more tasks than workers."
            )

        # Create the chunks with max_chunk_size
        worker_chunks = [
            [(i, force_profiles[i]) for i in all_indices[start:start + chunk_size]]
            for start in range(0, num_samples, chunk_size)
        ]

        init_args = (
            self.empty_force_profile,
            self.initial_density,
            self.time_steps,
            self.dt,
            self.parameters,
        )
        results = []

        ctx = get_context("fork")
        with ProcessPoolExecutor(
            mp_context=ctx,
            max_workers=num_workers,
            initializer=init_worker,
            initargs=init_args,
        ) as executor:
            futures = [executor.submit(run_worker_batches, chunk) for chunk in worker_chunks]

            completed_samples = 0
            for fut in as_completed(futures):
                batch_results = fut.result()
                results.extend(batch_results)
                completed_samples += len(batch_results)
                pct = completed_samples / num_samples * 100
                bar = "#" * int(pct // 2) + "." * (50 - int(pct // 2))
                logging.info(f"Progress: [{bar}] {pct:5.1f}%")

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

    generator = TrainingDataGenerator(
        output_dir="/home/gijs/Desktop/Thesis/data/raw",
        initial_density=initial_density,
        force_max=10,
        force_count_max=5,
        batch_seed=0,
    )
    start_time = time.time()
    generator.generate_parallel(1000)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logging.info(f"Simulation completed in {elapsed_time:.2f} seconds.")
