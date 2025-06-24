"""Splitting module for bone remodeling surrogate model.

This module provides functionality to split input and output features into training, validation, and test sets
without relying on external libraries such as scikit-learn. The main function, `splitting`, ensures reproducibility,
validates input parameters, and returns a NamedTuple containing the split datasets.

Usage:
    from splitter import splitting
    split_data = splitting(input_features, output_features, random_state=42)
"""

from typing import NamedTuple

import numpy as np

RANDOM_STATE_MAX = 10000


class SplitData(NamedTuple):
    """Named tuple to hold split data."""

    training_input: np.ndarray
    validation_input: np.ndarray
    test_input: np.ndarray
    training_output: np.ndarray
    validation_output: np.ndarray
    test_output: np.ndarray

def splitting(
    input_features: np.ndarray,
    output_features: np.ndarray,
    random_state: int | None = None,
    train_data_ratio: float | int = 0.7,
    validation_data_ratio: float | int = 0.15,
) -> SplitData:
    """Split data into training, validation, and test sets without using sklearn.

    Args:
    ----
        input_features (np.ndarray): Input features to be split.
        output_features (np.ndarray): Target labels to be split.
        random_state (int, optional): Random seed for reproducibility. If None, a random integer seed is generated using np.random.randint(0, RANDOM_STATE_MAX) (default is None).
        train_data_ratio (float | int): Proportion of data to use for training (default is 0.7).
        validation_data_ratio (float | int): Proportion of data to use for validation (default is 0.15).

    Raises:
    ------
        ValueError: If train_data_ratio, or validation_data_ratio are not floats between 0 and 1.
        ValueError: If input_features and output_features are not numpy arrays or have different lengths.
        ValueError: If input_features or output_features is empty.

    Returns:
    -------
        SplitData: NamedTuple containing split data (training_input, validation_input, test_input, training_output, validation_output, test_output).

    """
    seed = random_state if random_state is not None else np.random.randint(0, RANDOM_STATE_MAX)
    rng = np.random.default_rng(seed)

    # Cast ratios to float for user convenience and validate types
    try:
        train_data_ratio = float(train_data_ratio)
        validation_data_ratio = float(validation_data_ratio)
    except (TypeError, ValueError) as err:
        raise ValueError("train_data_ratio and validation_data_ratio must be convertible to float.") from err
    if not (0 < train_data_ratio < 1):
        raise ValueError("train_data_ratio must be between 0 and 1 (exclusive).")
    if not (0 < validation_data_ratio < 1):
        raise ValueError("validation_data_ratio must be between 0 and 1 (exclusive).")
    # Due to floating point arithmetic, the sum of ratios may slightly exceed 1; np.finfo(float).eps accounts for this precision issue.
    if (train_data_ratio + validation_data_ratio) >= 1 - np.finfo(float).eps:
        raise ValueError("The sum of train_data_ratio and validation_data_ratio must be less than 1 (considering floating point precision).")
    if not isinstance(input_features, np.ndarray) or not isinstance(output_features, np.ndarray):
        raise ValueError("Input features and output features must be numpy arrays.")
    if input_features.shape[0] != output_features.shape[0]:
        raise ValueError("Input features and output features must have the same number of samples (rows).")
    if input_features.shape[0] == 0 or output_features.shape[0] == 0:
        raise ValueError("Input features and output features cannot be empty.")

    # Create a random permutation of indices
    indices = np.arange(len(input_features))
    rng.shuffle(indices)
    n_total = len(indices)
    n_train = int(np.floor(train_data_ratio * n_total))
    n_val = int(np.floor(validation_data_ratio * n_total))

    # Split the data
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    training_input = input_features[train_indices]
    validation_input = input_features[val_indices]
    test_input = input_features[test_indices]

    training_output = output_features[train_indices]
    validation_output = output_features[val_indices]
    test_output = output_features[test_indices]

    return SplitData(
        training_input=training_input,
        validation_input=validation_input,
        test_input=test_input,
        training_output=training_output,
        validation_output=validation_output,
        test_output=test_output
    )
