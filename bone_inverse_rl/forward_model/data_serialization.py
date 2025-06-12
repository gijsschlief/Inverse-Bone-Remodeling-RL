import numpy as np
from typing import Dict

def serialize_data(
    serial_number: int, 
    force_profile: np.ndarray, 
    result: np.ndarray, 
    error: str = None) -> Dict:
    """
    Serialize the data into a dictionary format for saving or further processing.
    If there is no error, exclude the error field from the dictionary.
    """
    serialized_data = {
        "serial_number": serial_number,
        "force_profile": force_profile.tolist(),
        "final_output_density": result.tolist(),
    }
    if error is not None:
        serialized_data["error"] = error
    return serialized_data