"""Data reader for forward model data.

This module provides functionality to read and parse forward model data from JSON files.
It supports reading a single file, multiple files, or all JSON files in a directory.
The data is expected to be in a specific format, and the module converts it into NumPy arrays
for further processing.

Example usage:
    >>> from bone_remodeling.forward_model.data_reader import forward_data_reader
    >>> data = forward_data_reader("path/to/data.json")
    >>> if data:
    ...     serial_numbers, force_profiles, final_output_densities = data
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def forward_data_reader(
    file_path: Union[Path, str, Sequence[Union[str, Path]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read and parse forward model data from a JSON file or multiple JSON files in a directory.

    Args:
        file_path (Path | str | list[str | Path]): Path to the JSON file or a list of paths to JSON files.

    Returns:
        Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
            A tuple containing three NumPy arrays:
            - serial_numbers: Array of serial numbers.
            - force_profiles: Array of force profiles.
            - final_output_densities: Array of final output densities.

    Raises:
        ValueError: If the file path is not valid or if the file format is incorrect.

    """
    if not isinstance(file_path, (str, Path, list)):
        logging.error(
            f"Invalid file path type: {type(file_path)}. Expected str, Path, or list[str].",
        )
        return None
    if isinstance(file_path, list):
        if not all(isinstance(fp, (str, Path)) for fp in file_path):
            logging.error("All items in the list must be of type str or Path.")
            return None
        file_paths: list[Path] = [
            Path(fp) if not isinstance(fp, Path) else fp for fp in file_path
        ]
        return _forward_data_load_multiple(file_paths)
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if file_path.is_file():
        return _forward_data_load_single(file_path)
    if file_path.is_dir():
        file_list = sorted(file_path.glob("*.json"))
        if not file_list:
            logging.error(f"No JSON files found in directory: {file_path}")
            return None
        return _forward_data_load_multiple(file_list)

    logging.error(
        f"Invalid file path: {file_path}. It must be a file or a directory containing JSON files.",
    )
    return None


def _forward_data_load_multiple(
    file_paths: Sequence[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read and parse multiple JSON files containing forward model data.

    Args:
        file_paths (list[Path]): list of paths to the JSON files.

    Returns:
        Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
            A tuple containing three NumPy arrays:
            - serial_numbers: Array of serial numbers.
            - force_profiles: Array of force profiles.
            - final_output_densities: Array of final output densities.

    """
    all_entries = []

    for file_path in file_paths:
        try:
            with file_path.open("r") as f:
                file_data = json.load(f)
                all_entries.extend(file_data)
                logging.info(f"Loaded {len(file_data)} entries from {file_path.name}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Failed to load {file_path.name}: {e}")
            continue

    if not all_entries:
        logging.error("No valid data found in the provided files.")
        return None

    logging.info(
        f"In total loaded {len(all_entries)} entries from {len(file_paths)} files.",
    )
    return _convert_forward_data_to_numpy(all_entries)


def _forward_data_load_single(
    data_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read and parse a single JSON file containing forward model data.

    Args:
        data_path (Path): Path to the JSON file.

    Returns:
        Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
            A tuple containing three NumPy arrays:
            - serial_numbers: Array of serial numbers.
            - force_profiles: Array of force profiles.
            - final_output_densities: Array of final output densities.

    """
    if not data_path.is_file():
        logging.error(f"File does not exist: {data_path}")
        return None
    if data_path.suffix != ".json":
        logging.error(f"Invalid file format: {data_path}. Expected a .json file.")
        return None
    if not data_path.is_absolute():
        data_path = data_path.resolve()

    try:
        with data_path.open("r") as file:
            data = json.load(file)
        return _convert_forward_data_to_numpy(data)

    except FileNotFoundError:
        logging.exception(f"File not found: {data_path}")
    except json.JSONDecodeError:
        logging.exception(f"JSON decode error in file: {data_path}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
    return None


def _convert_forward_data_to_numpy(
    data: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of dictionaries containing forward model data into NumPy arrays.

    This function extracts serial numbers, force profiles, and final output densities
    from the provided data. It handles potential errors in the data format and logs
    any issues encountered during the extraction process.

    Args:
        data (list[dict[str, Any]]): list of dictionaries containing forward model data.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing three NumPy arrays:
            - serial_numbers: Array of serial numbers.
            - force_profiles: Array of force profiles.
            - final_output_densities: Array of final output densities.

    """
    serial_numbers = []
    force_profiles = []
    final_output_densities = []

    error_counts: dict[str, int] = {}
    for entry in data:
        try:
            serial_number = entry["serial_number"]
            if isinstance(serial_number, list):
                serial_numbers.append(serial_number[0])
            else:
                serial_numbers.append(serial_number)
            force_profiles.append(entry["force_profile"])
            final_output_densities.append(entry["final_output_density"])
        except (KeyError, IndexError, TypeError) as e:
            error_msg = str(e)
            error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
            continue

    for error_msg, count in error_counts.items():
        logging.warning(
            f"Skipped {count} malformed entr{'y' if count == 1 else 'ies'} (e.g., {error_msg})",
        )

    force_profiles_array = np.array(force_profiles, dtype=np.float32)
    final_densities_array = np.array(final_output_densities, dtype=np.float32)
    serial_numbers_array = np.array(serial_numbers)

    return serial_numbers_array, force_profiles_array, final_densities_array


# Example usage:
if __name__ == "__main__":
    directory_path = Path("/home/gijs/Desktop/Thesis/data/raw/")
    result = forward_data_reader(directory_path)
    if result is not None:
        _, force_profiles, output_densities = result
        logging.info(f"Force Profiles Shape: {force_profiles.shape}")
        logging.info(f"Output Densities Shape: {output_densities.shape}")
        logging.info(f"Force Profiles Sample: {force_profiles[0]}")
        logging.info(f"Output Densities Sample: {output_densities[0]}")
    else:
        logging.error("Failed to load data: forward_data_reader returned None.")
