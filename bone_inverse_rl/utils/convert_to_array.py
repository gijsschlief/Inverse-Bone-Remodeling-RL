from typing import List, Tuple, Dict, Any
import numpy as np

def convert_to_array(data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        force_profiles.append(np.array(entry['force_profile']).flatten())
        final_output_densities.append(np.array(entry['final_output_density']).flatten())

    return np.array(serial_numbers), np.array(force_profiles), np.array(final_output_densities)