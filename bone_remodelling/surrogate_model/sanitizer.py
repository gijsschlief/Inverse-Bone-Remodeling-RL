"""Sanitizer module for filtering out NaN values from predicted and true matrices."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def sanitize_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sanitize the input data by filtering out rows with NaN values in either x_data or y_data.

    Args:
    ----
        x_data (np.ndarray): Input features to be sanitized.
        y_data (np.ndarray): Target labels to be sanitized.

    Returns:
    -------
        tuple[np.ndarray, np.ndarray]: Sanitized x_data and y_data with NaN rows removed.

    Raises:
    ------
        ValueError: If the input data is not a numpy array.

    """
    if not isinstance(x_data, np.ndarray) or not isinstance(y_data, np.ndarray):
        raise ValueError("Both x_data and y_data must be numpy arrays.")
    # Flatten all but the first dimension to check for NaNs per sample
    x_mask = ~np.isnan(x_data.reshape(x_data.shape[0], -1)).any(axis=1)
    y_mask = ~np.isnan(y_data.reshape(y_data.shape[0], -1)).any(axis=1)
    mask = x_mask & y_mask
    removed = x_data.shape[0] - np.count_nonzero(mask)
    if removed > 0:
        logger.error(f"sanitize_data: Removed {removed} datapoints due to NaN values.")
    return x_data[mask], y_data[mask]
