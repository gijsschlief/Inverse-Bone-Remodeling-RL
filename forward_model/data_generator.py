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
from typing import Any, Dict, Optional, Union

import numpy as np
from bone_remodeling.forward_model.density_simulation import (
    DensitySimulation,  # type: ignore
)
from bone_remodeling.forward_model.force_profile_generator import (
    ForceProfileGenerator,  # type: ignore
)
from fenics import LogLevel, set_log_level  # type: ignore

set_log_level(LogLevel.ERROR)

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
        try:
            _worker_sim.reset()
            _worker_sim.update_force_profile(profile)
            _worker_sim.run()
            dens = _worker_sim.get_density()
            results.append(
                {
                    "serial_number": i + 1,
                    "force_profile": profile.tolist(),
                    "final_output_density": dens.tolist(),
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
    return results


class TrainingDataGenerator:
    """Class for generating training data for bone remodeling simulations."""

    def __init__(
        self,
        force_profiles: np.ndarray,
        output_dir: str,
        initial_density: Union[np.ndarray, None] = None,
        time_steps: int = 100,
        dt: float = 1.0,
    ) -> None:
        """Initialize the TrainingDataGenerator."""
        self.force_profiles: np.ndarray = force_profiles
        self.output_dir: Path = Path(output_dir)
        self.time_steps = time_steps
        self.dt = dt
        self.parameters: Dict[str, Any] | None = self._load_parameters()
        self.num_samples = force_profiles.shape[0]

        initial_density_value = (
            self.parameters.get("initial_density_value", 0.8)
            if self.parameters is not None
            else 0.8
        )

        if initial_density is None:
            sim_size = self.force_profiles.shape[2]
            self.initial_density: np.ndarray = np.full(
                (sim_size, sim_size), initial_density_value
            )
        else:
            self.initial_density: np.ndarray = initial_density
        self.empty_force_profile = np.zeros((3, np.max(self.initial_density.shape)))

        self._validate_input()
        os.makedirs(self.output_dir, exist_ok=True)

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
        """Validate the input parameters."""
        if not isinstance(self.force_profiles, np.ndarray):
            raise ValueError("force_profiles must be a numpy ndarray.")
        if not isinstance(self.output_dir, Path):
            raise ValueError("output_dir must be a Path object.")
        if not self.output_dir.exists():
            raise ValueError(f"Output directory {self.output_dir} does not exist.")
        if not isinstance(self.time_steps, int) or self.time_steps <= 0:
            raise ValueError("time_steps must be a positive integer.")
        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            raise ValueError("dt must be a positive number.")
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a dictionary.")
        if not isinstance(self.initial_density, np.ndarray):
            raise ValueError("initial_density must be a numpy ndarray.")

    def generate_parallel(
        self, max_chunk_size: int = 100, force_profile_name: str = "Undefined"
    ) -> list[dict]:
        """Generate training data in parallel."""
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            "%m%d_%H%M"
        )
        filepath = (
            self.output_dir
            / f"training_{force_profile_name}_{self.num_samples}_samples_{timestamp}.json"
        )

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

        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Training data saved to {filepath}")
        return results

    def generate_serial(self) -> list[dict]:
        """Generate training data serially."""
        results = []
        simulation = DensitySimulation(
            force_profile=self.empty_force_profile,
            initial_density_field=self.initial_density,
            time_steps=self.time_steps,
            dt=self.dt,
            parameters=self.parameters,
        )
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
        return results

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
        num_samples=1000,
        force_max=15.0,
    )

    logging.info("Running forward model simulations...")

    data_generator = TrainingDataGenerator(
        force_profiles=force_profiles,
        output_dir="/home/gijs/Desktop/Thesis/data/raw",
        initial_density=initial_density,
        time_steps=250,
        dt=1.0,
    )
    start_time = time.time()
    _ = data_generator.generate_parallel(
        max_chunk_size=40, force_profile_name="combined"
    )
    # _ = data_generator.generate_serial()
    stop_time = time.time()

    elapsed_time = stop_time - start_time
    logging.info(f"Simulation completed in {elapsed_time:.2f} seconds.")
