import json
from typing import List, Any,  Dict, Tuple, Optional
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def forward_data_reader(file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Reads and parses JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]: 
            A tuple containing three NumPy arrays:
                                                   - serial_numbers: Array of serial numbers.
                                                   - force_profiles: Array of flattened force profiles.
                                                   - final_output_densities: Array of flattened final output densities.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
        Exception: For any other unexpected errors.
    """
    path = Path(file_path)
    if not path.is_file():
        logging.error(f"File does not exist: {file_path}")
        return None
    if not path.suffix == '.json':
        logging.error(f"Invalid file format: {file_path}. Expected a .json file.")
        return None
    if not path.is_absolute():
        logging.error(f"File path is not absolute: {file_path}")
        return None
    
    try:
        with path.open('r') as file:
            data = json.load(file)
        return _convert_forward_data_to_numpy(data)
    

    except FileNotFoundError:
        logging.error(f"File not found: {path}")
    except json.JSONDecodeError:
        logging.error(f"JSON decode error in file: {path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return None

def _convert_forward_data_to_numpy(data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a list of dictionaries containing serial numbers, force profiles, 
    and final output densities into separate NumPy arrays.

    Args:
        data (List[Dict[str, Any]]): A list of dictionaries where each dictionary 
                                     contains the keys 'serial_number', 'force_profile', 
                                     and 'final_output_density'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three NumPy arrays:
                                                   - serial_numbers: Array of serial numbers.
                                                   - force_profiles: Array of flattened force profiles.
                                                   - final_output_densities: Array of flattened final output densities.
    """
    serial_numbers = []
    force_profiles = []
    final_output_densities = []

    for entry in data:
        serial_numbers.append(entry['serial_number'][0])
        force_profiles.append(entry['force_profile'])
        final_output_densities.append(entry['final_output_density'])

    force_profiles_array = np.array(force_profiles, dtype=np.float32).reshape(len(force_profiles), -1)
    final_densities_array = np.array(final_output_densities, dtype=np.float32).reshape(len(final_output_densities), -1)
    serial_numbers_array = np.array(serial_numbers)

    return serial_numbers_array, force_profiles_array, final_densities_array

# Example usage:
if __name__ == "__main__":
    FILE_PATH = "/home/gijs/Desktop/Thesis/data/raw/training_data_test.json"
    _, force_profiles, output_densities = forward_data_reader(FILE_PATH)
    logging.info(f"Force Profiles Shape: {force_profiles.shape}")
    logging.info(f"Output Densities Shape: {output_densities.shape}")
    logging.info(f"Force Profiles Sample: {force_profiles[0]}")
    logging.info(f"Output Densities Sample: {output_densities[0]}")