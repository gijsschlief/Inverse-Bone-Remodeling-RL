"""Module provides functions for normalizing and unnormalizing datasets."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def normalize_data(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize datasets based on training statistics.

    Args:
    ----
        data (np.ndarray): Dataset to normalize.
        mean (np.ndarray | None): Mean used for normalization, can be None.
        std (np.ndarray | None): Standard deviation used for normalization, can be None.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Normalized data, mean, and standard deviation.

    """
    if mean is None:
        mean = data.mean(axis=0, keepdims=True)
    if std is None:
        std = data.std(axis=0, keepdims=True) + 1e-8  # avoid division by zero
    normalized_data = (data - mean) / std
    return normalized_data, mean, std


def unnormalize_data(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Unnormalize data using the provided mean and standard deviation.

    Args:
    ----
        data (np.ndarray): Data to be unnormalized.
        mean (np.ndarray): Mean used for normalization.
        std (np.ndarray): Standard deviation used for normalization.

    Returns:
    -------
        np.ndarray: Unnormalized data.

    """
    if isinstance(data, np.ndarray):
        unnormalized_data = data * std + mean
        return unnormalized_data.astype(np.float32)
    elif isinstance(data, torch.Tensor):
        tensor = data.float()
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=tensor.device)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=tensor.device)
    return tensor * std_tensor + mean_tensor


def save_normalization_params(
    path: Path,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> None:
    """Save normalization parameters to a file.

    Args:
    ----
        path (Path): Path to the .npz file where normalization parameters will be saved.
        x_mean (np.ndarray): Mean of the input features.
        x_std (np.ndarray): Standard deviation of the input features.
        y_mean (np.ndarray): Mean of the target variable.
        y_std (np.ndarray): Standard deviation of the target variable.

    Raises:
    ------
        ValueError: If the path is not a valid .npz file or if saving fails

    """
    try:
        np.savez(path, X_mean=x_mean, X_std=x_std, y_mean=y_mean, y_std=y_std)
    except Exception as e:
        logging.error(f"Failed to save normalization parameters: {e}")
        raise ValueError(f"Failed to save normalization parameters to {path}") from e


def load_normalization_params(
    path: Path | str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load normalization parameters from a file.

    Args:
    ----
        path (Path | str): Path to the .npz file containing normalization parameters.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_mean, X_std, y_mean, y_std loaded from the file.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Normalization parameters file not found: {path}")
    if not path.suffix == ".npz":
        raise ValueError(f"Expected a .npz file, got {path.suffix}")
    if not path.is_file():
        raise ValueError(f"Expected a file, but found a directory: {path}")
    if not path.stat().st_size > 0:
        raise ValueError(f"File is empty: {path}")
    logging.info(f"Loading normalization parameters from {path}")
    data = np.load(path)

    x_mean = data.get("X_mean")
    x_std = data.get("X_std")
    y_mean = data.get("y_mean")
    y_std = data.get("y_std")
    return x_mean, x_std, y_mean, y_std
