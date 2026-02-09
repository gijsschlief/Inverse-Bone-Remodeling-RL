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
