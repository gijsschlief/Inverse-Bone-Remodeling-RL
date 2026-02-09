"""Module for converting between triangular force profiles and their parameterized representations (peak location, peak side, peak height)."""

import numpy as np


def force_profile_to_params(force_profile: np.ndarray) -> tuple[int, int, float]:
    """Convert a 3xN force profile into peak location, peak side, and peak height."""
    total_sides = 3
    if force_profile.shape[0] != total_sides:
        raise ValueError("Force profile must have shape (3, N)")

    peak_side = int(np.argmax(np.max(force_profile, axis=1)))
    peak_height = float(np.max(force_profile[peak_side]))
    peak_location = int(np.argmax(force_profile[peak_side]))

    return peak_location, peak_side, peak_height

def params_to_force_profile(
    peak_location: int,
    peak_side: int,
    peak_height: float,
    length: int = 10,
) -> np.ndarray:
    """Convert peak location, peak side, and peak height back into a triangular 3xN force profile."""
    peak_side = min(peak_side, 2)
    total_sides = 3
    if not (0 <= peak_side < total_sides):
        raise ValueError("Peak side must be 0, 1, or 2")
    if not (0 <= peak_location < length):
        raise ValueError(f"Peak location must be between 0 and {length - 1}")

    force_profile = np.zeros((3, length), dtype=np.float32)

    for j in range(length):
        if j < peak_location and peak_location > 0:
            force_profile[peak_side, j] = peak_height * (j / peak_location)
        elif j > peak_location and peak_location < length - 1:
            force_profile[peak_side, j] = peak_height * ((length - 1 - j) / (length - 1 - peak_location))
        else:
            force_profile[peak_side, j] = peak_height
    return force_profile

def reshape_input_features_for_model(unconverted_data: np.ndarray) -> np.ndarray:
    """Convert new data to tensor format for the model.

    Args:
        unconverted_data (np.ndarray): Input features (N, int, int, float).

    Returns:
        converted_data (np.ndarray): Converted tensor data reshaped for the model.

    """
    converted_data = np.zeros((unconverted_data.shape[0], 31), dtype=np.float32)
    for i in range(unconverted_data.shape[0]):
        side = unconverted_data[i][1]  # peak side
        location = unconverted_data[i][0]  # peak location
        magnitude = unconverted_data[i][2]  # peak height
        class_index = int(side * 10 + location)
        force_location_vector = np.zeros(30, dtype=np.float32)
        force_location_vector[class_index] = 100.0
        converted_data[i] = np.concatenate([force_location_vector, [magnitude]])
    return converted_data
