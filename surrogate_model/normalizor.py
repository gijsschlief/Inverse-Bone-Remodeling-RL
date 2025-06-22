"""Module provides functions for normalizing and unnormalizing datasets."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def normalize_data(train: np.ndarray, val: np.ndarray, test: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize datasets based on training statistics.

    Args:
    ----
        train (np.ndarray): Training dataset.
        val (np.ndarray): Validation dataset.
        test (np.ndarray): Test dataset.
        mean (np.ndarray | None): Precomputed mean for normalization. If None, it will be computed from the training data.
        std (np.ndarray | None): Precomputed standard deviation for normalization. If None, it will be computed from the training data.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Normalized training, validation, and test datasets, along with the mean and standard deviation used for normalization.

    """
    if mean is None:
        mean = train.mean(axis=0, keepdims=True)
    if std is None:
        std = train.std(axis=0, keepdims=True) + 1e-8  # avoid division by zero
    normalized_train = (train - mean) / std
    normalized_val = (val - mean) / std
    normalized_test = (test - mean) / std
    return normalized_train, normalized_val, normalized_test, mean, std

def unnormalize_data(data: torch.Tensor | np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor | np.ndarray:
    """Unnormalize data using the provided mean and standard deviation.

    Args:
    ----
        data (torch.Tensor | np.ndarray): Data to be unnormalized.
        mean (np.ndarray): Mean used for normalization.
        std (np.ndarray): Standard deviation used for normalization.

    Returns:
    -------
        torch.Tensor | np.ndarray: Unnormalized data.

    """
    if isinstance(data, np.ndarray):
        unnormalized_data = data * std + mean
        return unnormalized_data.astype(np.float32)
    elif isinstance(data, torch.Tensor):
        tensor = data.float()
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=tensor.device)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=tensor.device)
    return tensor * std_tensor + mean_tensor

def save_normalization_params(path: Path, X_mean: np.ndarray, X_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> None:
    """Save normalization parameters to a file.

    Args:
    ----
        path (Path): Path to the .npz file where normalization parameters will be saved.
        X_mean (np.ndarray): Mean of the input features.
        X_std (np.ndarray): Standard deviation of the input features.
        y_mean (np.ndarray): Mean of the target variable.
        y_std (np.ndarray): Standard deviation of the target variable.

    Raises:
    ------
        ValueError: If the path is not a valid .npz file or if saving fails

    """
    try:
        np.savez(path, X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
    except Exception as e:
        logging.error(f"Failed to save normalization parameters: {e}")
        raise ValueError(f"Failed to save normalization parameters to {path}") from e

def load_normalization_params(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load normalization parameters from a file.

    Args:
    ----
        path (Path): Path to the .npz file containing normalization parameters.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_mean, X_std, y_mean, y_std loaded from the file.

    """
    if not path.exists():
        raise FileNotFoundError(f"Normalization parameters file not found: {path}")
    if not path.suffix == '.npz':
        raise ValueError(f"Expected a .npz file, got {path.suffix}")
    if not path.is_file():
        raise ValueError(f"Expected a file, but found a directory: {path}")
    if not path.stat().st_size > 0:
        raise ValueError(f"File is empty: {path}")
    logging.info(f"Loading normalization parameters from {path}")
    data = np.load(path)

    X_mean = data.get('X_mean')
    X_std = data.get('X_std')
    y_mean = data.get('y_mean')
    y_std = data.get('y_std')
    return X_mean, X_std, y_mean, y_std
