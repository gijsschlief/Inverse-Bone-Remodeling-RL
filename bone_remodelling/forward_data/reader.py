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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def forward_data_reader(  # noqa: PLR0911
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
        logger.error(
            f"Invalid file path type: {type(file_path)}. Expected str, Path, or list[str].",
        )
        return None

    if isinstance(file_path, list):
        if not all(isinstance(fp, (str, Path)) for fp in file_path):
            logger.error("All items in the list must be of type str or Path.")
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
            logger.error(f"No JSON files found in directory: {file_path}")
            return None
        return _forward_data_load_multiple(file_list)

    logger.error(
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
                logger.info(f"Loaded {len(file_data)} entries from {file_path.name}")
        except (FileNotFoundError, json.JSONDecodeError) as e:  # noqa: PERF203
            logger.warning(f"Failed to load {file_path.name}: {e}")
            continue

    if not all_entries:
        logger.error("No valid data found in the provided files.")
        return None

    logger.info(
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
        logger.error(f"File does not exist: {data_path}")
        return None
    if data_path.suffix != ".json":
        logger.error(f"Invalid file format: {data_path}. Expected a .json file.")
        return None
    if not data_path.is_absolute():
        data_path = data_path.resolve()

    try:
        with data_path.open("r") as file:
            data = json.load(file)
        return _convert_forward_data_to_numpy(data)

    except FileNotFoundError:
        logger.exception(f"File not found: {data_path}")
    except json.JSONDecodeError:
        logger.exception(f"JSON decode error in file: {data_path}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
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
        except (KeyError, IndexError, TypeError) as e:  # noqa: PERF203
            error_msg = str(e)
            error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
            continue

    for error_msg, count in error_counts.items():
        logger.warning(
            f"Skipped {count} malformed entr{'y' if count == 1 else 'ies'} (e.g., {error_msg})",
        )

    force_profiles_array = np.array(force_profiles, dtype=np.float32)
    final_densities_array = np.array(final_output_densities, dtype=np.float32)
    serial_numbers_array = np.array(serial_numbers)

    return serial_numbers_array, force_profiles_array, final_densities_array

################################ ADDITIONAL CODE FOR DIVERSITY METRICS ################################

def compute_diversity_metrics(force_profiles: np.ndarray) -> dict[str, float]:
    """Compute simple dataset diversity metrics for biomechanical force profiles."""
    # Coverage fraction (fraction of nodes ever loaded)
    nonzero_counts = np.count_nonzero(force_profiles, axis=0)
    coverage_fraction = np.count_nonzero(nonzero_counts) / force_profiles.shape[1]

    # Shannon entropy of location distribution (normalized)
    # Average absolute load per node, normalised to sum 1
    node_loads = np.abs(force_profiles).sum(axis=0)
    normalized_node_loads = node_loads / node_loads.sum()
    location_entropy = entropy(normalized_node_loads)
    max_entropy = np.log(len(normalized_node_loads))
    normalized_entropy = location_entropy / max_entropy

    # Mean pairwise L2 distance (sampled for efficiency)
    n_samples = min(2000, force_profiles.shape[0])
    sample_indices = np.random.choice(force_profiles.shape[0], n_samples, replace=False)
    sampled = force_profiles[sample_indices]
    pairwise_differences = sampled[:, None, :] - sampled[None, :, :]
    l2 = np.linalg.norm(pairwise_differences, axis=-1)
    mean_pairwise_distance = np.mean(l2[np.triu_indices_from(l2, k=1)])

    # Force energy variance
    energies = np.sum(force_profiles**2, axis=1)
    energy_mean = np.mean(energies)
    energy_var = np.var(energies)

    return {
        "coverage_fraction": coverage_fraction,
        "normalized_entropy": normalized_entropy,
        "mean_pairwise_distance": mean_pairwise_distance,
        "energy_mean": energy_mean,
        "energy_var": energy_var,
    }


if __name__ == "__main__":
    directory_path = Path("/home/gijs/Desktop/Thesis/data/raw/")
    result = forward_data_reader(directory_path)
    if result is not None:
        _, force_profiles, output_densities = result
        logger.info(f"Force Profiles Shape: {force_profiles.shape}")
        logger.info(f"Output Densities Shape: {output_densities.shape}")
        logger.info(f"Force Profiles Sample: {force_profiles[0]}")
        logger.info(f"Output Densities Sample: {output_densities[0]}")
    else:
        logger.error("Failed to load data: forward_data_reader returned None.")

    # Add a plot that visualizes the distribution of the output densities
    import matplotlib.pyplot as plt
    from scipy.stats import entropy  # type: ignore

    from bone_remodelling.forward_model.density_visualizer import (
        plot_density_matrix,
    )
    avg_output_densities = output_densities.mean(axis=0)
    avg_force_profiles = force_profiles.mean(axis=0)
    std_force_profiles = force_profiles.std(axis=0)
    std_output_densities = output_densities.std(axis=0)

    plot_density_matrix(
        matrix=avg_output_densities,
        force_profile=avg_force_profiles,
        title="Location average of output densities - SL dataset",
        axis=plt.gca(),
    )
    plt.show()

    directory_path_triangular = Path("/home/gijs/Desktop/Thesis/data/raw/triangular/")
    result_triangular = forward_data_reader(directory_path_triangular)
    if result_triangular is not None:
        _, force_profiles_triangular, output_densities_triangular = result_triangular
    else:
        logger.error("Failed to load data from triangular dataset: forward_data_reader returned None.")

    force_profile_energy = np.zeros(force_profiles.shape[0])
    force_profile_flat = np.zeros((force_profiles.shape[0], force_profiles.shape[1]*force_profiles.shape[2]))
    for i in range(force_profiles.shape[0]):
        force_profile_energy[i] = np.square(force_profiles[i]).sum()
        force_profile_flat[i] = force_profiles[i].flatten()

    force_profile_energy_triangular = np.zeros(force_profiles_triangular.shape[0])
    force_profile_flat_triangular = np.zeros((force_profiles_triangular.shape[0], force_profiles_triangular.shape[1]*force_profiles_triangular.shape[2]))
    for i in range(force_profiles_triangular.shape[0]):
        force_profile_energy_triangular[i] = np.square(force_profiles_triangular[i]).sum()
        force_profile_flat_triangular[i] = force_profiles_triangular[i].flatten()

    plt.figure(1)
    plt.hist(force_profiles.flatten(), bins=200)
    plt.hist(force_profiles_triangular.flatten(), bins=200)
    plt.title("Distribution of Force Profiles")
    plt.yscale("log")
    plt.xlabel("Force Profile Value")
    plt.ylabel("Frequency")
    plt.legend(["Dataset Supervised Learning", "Dataset RL"])

    plt.figure(2)
    plt.hist(output_densities.flatten(), bins=200)
    plt.hist(output_densities_triangular.flatten(), bins=200)
    plt.title("Distribution of Output Densities")
    plt.yscale("log")
    plt.xlabel("Output Density Value")
    plt.ylabel("Frequency")
    plt.legend(["Dataset Supervised Learning", "Dataset RL"])

    plt.figure(3)
    plt.hist(force_profile_energy, bins=200)
    plt.hist(force_profile_energy_triangular, bins=200)
    plt.title("Distribution of Force Profile Energy")
    plt.yscale("log")
    plt.xlabel("Force Profile Energy Value")
    plt.ylabel("Frequency")
    plt.legend(["Dataset Supervised Learning", "Dataset RL"])
    plt.show()


    metrics_supervised = compute_diversity_metrics(force_profile_flat)
    metrics_rl = compute_diversity_metrics(force_profile_flat_triangular)

    logger.info("Metric | Supervised | RL")
    logger.info("--------------------------------------")
    for key in metrics_supervised:
        logger.info(f"{key:25s} | {metrics_supervised[key]:8.3f} | {metrics_rl[key]:8.3f}")
