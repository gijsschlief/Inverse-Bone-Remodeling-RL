import numpy as np
from typing import Tuple

def split_data(
    force_profiles: np.ndarray,
    final_output_densities: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data into training, validation, and test sets without using sklearn.

    Parameters:
        force_profiles (np.ndarray): The input features.
        final_output_densities (np.ndarray): The target labels.
        train_size (float): Proportion of the data to include in the training set.
        val_size (float): Proportion of the data to include in the validation set.
        test_size (float): Proportion of the data to include in the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Split data (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    assert train_size + val_size + test_size == 1, "Train, validation, and test sizes must sum to 1."

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Shuffle the indices
    indices = np.arange(len(force_profiles))
    np.random.shuffle(indices)

    # Calculate split indices
    train_end = int(train_size * len(indices))
    val_end = train_end + int(val_size * len(indices))

    # Split the data
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    X_train = force_profiles[train_indices]
    X_val = force_profiles[val_indices]
    X_test = force_profiles[test_indices]

    y_train = final_output_densities[train_indices]
    y_val = final_output_densities[val_indices]
    y_test = final_output_densities[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test