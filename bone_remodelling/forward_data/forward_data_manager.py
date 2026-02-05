"""Manager for forward data storage and retrieval."""

import json
import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

class ForwardDataManager:
    """Manager for storing and retrieving forward model data."""

    def __init__(self, storage_path: Path) -> None:
        """Initialize the ForwardDataManager with a storage path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.storage_path = Path(storage_path) / Path(f"sim_{timestamp}.jsonl")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.serial_number = 0

    def save_data(self, force_profiles: Iterable, final_densities: Iterable) -> None:
        """Save data to the JSONL file, appending to it if it already exists."""
        with self.storage_path.open("a") as f:
                    for force_profile, density in zip(force_profiles, final_densities):
                        entry = {
                            "serial_number": self.serial_number,
                            "force_profile": force_profile.tolist(),
                            "final_output_density": density.tolist(),
                        }
                        f.write(json.dumps(entry) + "\n")
                        self.serial_number += 1

    def load_file(self, file_path: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Read the entire dataset back into memory for training."""
        if file_path is None:
            file_path = self.storage_path
        if not file_path.exists():
            logger.error(f"No data found at {file_path}")
            return None

        entries = [json.loads(stripped) for line in file_path.open("r") if (stripped := line.strip())]
        if not entries:
            return None
        return self._convert_to_numpy(entries)

    def load_directory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Read and parse all JSON files in the storage directory."""
        if not self.storage_path.parent.exists():
                logger.error(f"Storage directory does not exist: {self.storage_path.parent}")
                return None

        file_paths = sorted(self.storage_path.parent.glob("sim_*.jsonl"))
        all_serials, all_forces, all_densities = [], [], []

        for file_path in file_paths:
            data = self.load_file(file_path)
            if data:
                serial, forces, densities = data
                all_serials.append(serial)
                all_forces.append(forces)
                all_densities.append(densities)

        if not all_serials:
            return None

        return (
            np.concatenate(all_serials),
            np.concatenate(all_forces),
            np.concatenate(all_densities),
        )

    def _convert_to_numpy(self, data: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        serials = np.array([e["serial_number"] for e in data])
        forces = np.array([e["force_profile"] for e in data], dtype=np.float32)
        densities = np.array([e["final_output_density"] for e in data], dtype=np.float32)
        return serials, forces, densities
