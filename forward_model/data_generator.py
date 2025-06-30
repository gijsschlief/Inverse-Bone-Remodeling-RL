"""Module that provides a class for generating data for bone remodeling simulations.

It includes methods for generating random force profiles, running simulations
in parallel or sequentially, and saving the results to JSON files.
"""

import datetime
import json
import logging
import os
from multiprocessing import Pool, cpu_count
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

# Global variable for per-worker simulation instance
_worker_sim: Optional[DensitySimulation] = None


def init_worker(force_profile: np.ndarray, initial_density: np.ndarray, time_steps: int, dt: float, parameters: Dict[str, Any]) -> None:
    """Initialize a global DensitySimulation instance for this worker process."""
    global _worker_sim
    _worker_sim = DensitySimulation(
        force_profile=force_profile,
        initial_density_field=initial_density,
        time_steps=time_steps,
        dt=dt,
        parameters=parameters,
    )
    _worker_sim.run()

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

    def _generate_random_force_profile(self, sample_index: int) -> np.ndarray:
        np.random.seed(self.batch_seed + sample_index)
        force_profile = np.zeros((3, np.max(self.initial_density.shape)))
        self.empty_force_profile = force_profile.copy()
        num_forces = np.random.randint(1, self.force_count_max)
        locations = np.random.choice(
            np.prod(force_profile.shape),
            num_forces,
            replace=False,
        )
        for loc in locations:
            row, col = divmod(loc, force_profile.shape[1])
            force_profile[row, col] = np.random.uniform(-self.force_max, self.force_max)
        return force_profile


    def _run_sample(self, args: Tuple[int, np.ndarray]) -> Dict:
        i, force_profile = args
        error: Optional[str] = None
        try:
            global _worker_sim
            if _worker_sim is None:
                raise RuntimeError("Worker simulation instance is not initialized.")
            _worker_sim.reset()
            _worker_sim.update_force_profile(force_profile)
            _worker_sim.run()
            output = _worker_sim.get_density()
        except Exception as ex:
            error = str(ex)
            logging.exception(
                f"Error in sample {i}: {error} | Force profile: {force_profile}"
            )
            output = np.full_like(self.initial_density, np.nan)  # Default output on error

        return self.serialize_data(
            serial_number=i + 1,
            force_profile=force_profile,
            result=output,
            error=error,
        )

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

        args = [(i, self._generate_random_force_profile(i)) for i in range(num_samples)]
        init_args = (self.empty_force_profile, self.initial_density, self.time_steps, self.dt, self.parameters)

        with Pool(processes=cpu_count(),
                  initializer=init_worker,
                  initargs=init_args,) as pool:
            results = []
            for idx, result in enumerate(
                pool.imap_unordered(self._run_sample, args),
                1,
            ):
                results.append(result)
                progress = (idx / num_samples) * 100
                logging.info(
                    f"Progress: [{'#' * int(progress // 2)}{'.' * (50 - int(progress // 2))}] {progress:.2f}%",
                )

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
        filepath = (
            self.output_dir
            / f"training_batch_{self.batch_seed}_samples_{num_samples}_{timestamp}.json"
        )

        data = []
        for i in range(num_samples):
            force_profile = self._generate_random_force_profile(i)
            start = datetime.datetime.now() if i == 1 else None
            data_point = self._run_sample((i, force_profile))
            end = datetime.datetime.now() if i == 11 else None

            if start and end:
                elapsed = (end - start).total_seconds() / 10
                logging.info(
                    f"Average time per sample (samples 2 till 11): {elapsed:.4f} seconds",
                )

            data.append(data_point)

            # Progress log
            progress = (i + 1) / num_samples * 100
            logging.info(
                f"Progress: [{'#' * int(progress // 2)}{'.' * (50 - int(progress // 2))}] {progress:.2f}%",
            )

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
