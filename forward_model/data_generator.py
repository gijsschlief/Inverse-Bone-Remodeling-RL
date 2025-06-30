"""Module that provides a class for generating data for bone remodeling simulations.

It includes methods for generating random force profiles, running simulations
in parallel or sequentially, and saving the results to JSON files.
"""

import datetime
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from bone_remodeling.forward_model.density_simulation import (
    DensitySimulation,  # type: ignore
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


_worker_sim       : DensitySimulation
_worker_rng       : np.random.Generator
_profile_length   : int
_force_count_max  : int
_force_max        : float


def init_worker(
    empty_profile: np.ndarray,
    initial_density: np.ndarray,
    time_steps: int,
    dt: float,
    parameters: dict,
    batch_seed: int,
    force_count_max: int,
    force_max: float,
) -> None:
    """Initialize the per‐process simulation and RNG."""
    global _worker_sim, _worker_rng, _profile_length
    global _force_count_max, _force_max

    _worker_sim = DensitySimulation(
        force_profile=empty_profile,
        initial_density_field=initial_density,
        time_steps=time_steps,
        dt=dt,
        parameters=parameters,
    )

    # one RNG per worker, seeded once
    _worker_rng      = np.random.default_rng(batch_seed)
    _profile_length  = empty_profile.shape[1]
    _force_count_max = force_count_max
    _force_max       = force_max

def run_sample(i: int) -> dict:
    """Generate one random force profile, run sim, return serialized result."""
    # 1) build random force_profile
    profile = np.zeros((3, _profile_length))
    count   = _worker_rng.integers(1, _force_count_max)
    flat    = _worker_rng.choice(3*_profile_length, size=count, replace=False)
    mags    = _worker_rng.uniform(-_force_max, _force_max, size=count)

    rows, cols = divmod(flat, _profile_length)
    profile[rows, cols] = mags

    # 2) run the simulation
    _worker_sim.reset()
    _worker_sim.update_force_profile(profile)
    _worker_sim.run()
    dens = _worker_sim.get_density()

    return {
        "serial_number": i+1,
        "force_profile": profile.tolist(),
        "final_output_density": dens.tolist(),
    }

class TrainingDataGenerator:
    """Class for generating training data for bone remodeling simulations.

    This class provides methods to generate random force profiles, run simulations
    in parallel or sequentially, and save the results to JSON files. It also includes
    functionality to generate edge cases for testing the robustness of the model.
    """

    def __init__(
        self,
        output_dir: str,
        initial_density: Union[np.ndarray, None] = None,
        force_max: int = 2,
        force_count_max: int = 7,
        batch_seed: int = 0,
    ) -> None:
        """Initialize the TrainingDataGenerator with parameters for generating training data.

        Parameters
        ----------
        output_dir : str
            Directory where the generated training data will be saved.
        initial_density : np.ndarray, optional
            Initial density array to use for the simulation. If None, a default
            10x10 array filled with 0.8 is used.
        force_max : int, optional
            Maximum force magnitude to apply in the simulation. Default is 2.
        force_count_max : int, optional
            Maximum number of forces to apply in each sample. Default is 7.
        batch_seed : int, optional
            Seed for random number generation to ensure reproducibility. Default is 0.

        Raises
        ------
        ValueError
            If the output directory does not exist, or if any of the parameters are invalid.

        """
        self.output_dir: Path = Path(output_dir)
        self.force_max: int = force_max
        self.force_count_max: int = force_count_max
        self.batch_seed: int = batch_seed
        self.time_steps: int = 100
        self.dt: float = 1.0
        self.initial_density: np.ndarray = (
            np.full((10, 10), 0.8) if initial_density is None else initial_density
        )
        self.parameters: Dict[str, Any] | None = self._load_parameters()
        self._validate_input()
        os.makedirs(self.output_dir, exist_ok=True)
        self.empty_force_profile = np.zeros((3, np.max(self.initial_density.shape)))
        self._rng = np.random.default_rng(batch_seed)
        self._profile_length = np.max(self.initial_density.shape)

    def _load_parameters(self) -> Dict:
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
        """Validate the output directory, maximum force, maximum force count and batch seed."""
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
        if not isinstance(self.batch_seed, int):
            raise ValueError("batch_seed must be an integer.")

    def _generate_random_force_profiles(self, num_samples: int) -> np.ndarray:
        """Efficiently generate the random force profiles."""
        profiles = np.zeros((num_samples, 3, self._profile_length), dtype=float)
        for i in range(num_samples):
            count = self._rng.integers(1, self.force_count_max)
            if count > 0:
                idx = self._rng.choice(profiles.shape[1] * profiles.shape[2], size=count, replace=False)
                profiles[i].flat[idx] = self._rng.uniform(-self.force_max, self.force_max, size=count)
        return profiles

    def generate_parallel(self, num_samples: int) -> None:
        """Generate training data samples in parallel and save them to a JSON file.

        Parameters
        ----------
        num_samples : int
            The number of training data samples to generate.

        """
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            "%m%d_%H%M"
        )
        filepath = (
            self.output_dir
            / f"training_batch_{self.batch_seed}_samples_{num_samples}_{timestamp}.json"
        )
        init_args = (self.empty_force_profile,
        self.initial_density,
        self.time_steps,
        self.dt,
        self.parameters,
        self.batch_seed,
        self.force_count_max,
        self.force_max)
        results = []

        ctx = get_context("fork")
        with ProcessPoolExecutor(
            mp_context=ctx,
            max_workers=cpu_count(),
            initializer=init_worker,
            initargs=init_args
        ) as executor:
            # submit all the jobs
            futures = [executor.submit(run_sample, i) for i in range(num_samples)]

            # as each finishes, collect and log progress
            for idx, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                results.append(res)
                pct = idx / num_samples * 100
                bar = "#" * int(pct // 2) + "." * (50 - int(pct // 2))
                logging.info(f"Progress: [{bar}] {pct:5.1f}%")

        with Path.open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Training data saved to {filepath}")

    def generate_sequential(self, num_samples: int) -> None:
        """Generate training data samples sequentially and save them to a JSON file.

        Parameters
        ----------
        num_samples : int
            The number of training data samples to generate.

        """
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        filepath = self.output_dir / f"training_batch_{self.batch_seed}_samples_{num_samples}_{timestamp}.json"

        # -- build a single simulation up front, with a dummy force_profile --
        empty_profile = np.zeros((3, np.max(self.initial_density.shape)))
        sim = DensitySimulation(
            force_profile=empty_profile,
            initial_density_field=self.initial_density,
            time_steps=self.time_steps,
            dt=self.dt,
            parameters=self.parameters,
        )

        data = []
        force_profiles = self._generate_random_force_profiles(num_samples)
        for i in range(num_samples):
            fp = force_profiles[i]

            # reset & rerun rather than re-construct
            sim.reset()
            sim.update_force_profile(fp)
            sim.run()
            density = sim.get_density()

            data.append(self.serialize_data(
                serial_number=i + 1,
                force_profile=fp,
                result=density,
            ))

            # progress logging
            pct = (i + 1) / num_samples * 100
            bar = "#" * int(pct // 2) + "." * (50 - int(pct // 2))
            logging.info(f"Progress: [{bar}] {pct:.2f}%")

            # occasionally flush partial JSON
            if (i + 1) % 100 == 0 or (i + 1) == num_samples:
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=4)

        logging.info(f"Training data saved to {filepath}")

    @staticmethod
    def serialize_data(
        serial_number: int,
        force_profile: np.ndarray,
        result: np.ndarray,
        error: Optional[str] = None,
    ) -> Dict:
        """Serialize the data into a dictionary format.

        Parameters
        ----------
        serial_number : int
            Serial number of the sample.
        force_profile : np.ndarray
            Force profile applied in the simulation.
        result : np.ndarray
            Final output density after the simulation.
        error : str, optional
            Error message if an error occurred during the simulation.

        Returns
        -------
        Dict
            Serialized data containing serial number, force profile, final output density,
            and error message if applicable.

        """
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

    from fenics import LogLevel, set_log_level
    set_log_level(LogLevel.ERROR)
    initial_density = np.full((10, 10), 0.8)

    generator = TrainingDataGenerator(
        output_dir="/home/gijs/Desktop/Thesis/data/raw",
        initial_density=initial_density,
        force_max=10,
        force_count_max=5,
        batch_seed=0,
    )
    start_time = time.time()
    #generator.generate_sequential(100)
    generator.generate_parallel(100)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logging.info(f"Simulation completed in {elapsed_time:.2f} seconds.")

