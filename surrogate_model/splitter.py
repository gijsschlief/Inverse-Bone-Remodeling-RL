"""Splitting module for bone remodeling surrogate model."""

from typing import Tuple

import numpy as np


def splitting(
    X: np.ndarray,
    Y: np.ndarray,
    random_state: int = np.random.randint(0, 10000),
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training, validation, and test sets without using sklearn.

    Args:
    ----
        X (np.ndarray): Input features.
        Y (np.ndarray): Target labels.
        train_size (float): Proportion of the dataset to include in the training set.
        val_size (float): Proportion of the dataset to include in the validation set.
        test_size (float): Proportion of the dataset to include in the test set.
        random_state (int): Random seed for reproducibility.

    Raises:
    ------
        ValueError: If train_size, val_size, or test_size are not floats between 0 and 1.
        AssertionError: If train_size, val_size, and test_size do not sum to 1.
        ValueError: If X and Y are not numpy arrays or have different lengths.
        ValueError: If X or Y is empty.

    Returns:
    -------
        tuple: Split data (X_train, X_val, X_test, y_train, y_val, y_test).

    """
    np.random.seed(random_state)

    # Validate all input parameters
    if (
        not isinstance(train_size, float)
        or not isinstance(val_size, float)
        or not isinstance(test_size, float)
    ):
        raise ValueError("train_size, val_size, and test_size must be floats.")

    if not (0 < train_size < 1) or not (0 < val_size < 1) or not (0 < test_size < 1):
        raise ValueError(
            "train_size, val_size, and test_size must be between 0 and 1 (exclusive)."
        )

    if train_size + val_size + test_size != 1:
        raise AssertionError("Train, validation, and test sizes must sum to 1.")

    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise ValueError("Input features and target labels must be numpy arrays.")

    if len(X) != len(Y):
        raise ValueError("Input features and target labels must have the same length.")

    if len(X) == 0 or len(Y) == 0:
        raise ValueError("Input features and target labels cannot be empty.")

    # Create a random permutation of indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Calculate split indices
    train_end = int(train_size * len(indices))
    val_end = train_end + int(val_size * len(indices))

    # Split the data
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]

    y_train = Y[train_indices]
    y_val = Y[val_indices]
    y_test = Y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test
