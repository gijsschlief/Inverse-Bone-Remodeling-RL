"""Tests for the SplitData class and splitting function in the bone_remodeling.surrogate_model.splitter module."""

import numpy as np
import pytest
from bone_remodeling.surrogate_model.splitter import SplitData, splitting


def make_data(n_samples: int = 100, n_features: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for testing."""
    x = np.arange(n_samples * n_features).reshape(n_samples, n_features)
    y = np.arange(n_samples * 3).reshape(n_samples, 3)[:, 0]  # Use only the first column for output
    return x, y

def test_split_shapes_and_types() -> None:
    """Test that the splitting function returns SplitData with correct shapes and types."""
    x, y = make_data(50, 3)
    split = splitting(x, y, random_state=123, train_data_ratio=0.6, validation_data_ratio=0.2)
    assert isinstance(split, SplitData)
    n_total = x.shape[0]
    n_train = int(np.floor(0.6 * n_total))
    n_val = int(np.floor(0.2 * n_total))
    n_test = n_total - n_train - n_val
    assert split.training_input.shape == (n_train, 3)
    assert split.validation_input.shape == (n_val, 3)
    assert split.test_input.shape == (n_test, 3)
    assert split.training_output.shape == (n_train,)
    assert split.validation_output.shape == (n_val,)
    assert split.test_output.shape == (n_test,)

def test_split_reproducibility() -> None:
    """Test that the splitting function is reproducible with the same random state."""
    x, y = make_data(30, 2)
    split1 = splitting(x, y, random_state=42)
    split2 = splitting(x, y, random_state=42)
    np.testing.assert_array_equal(split1.training_input, split2.training_input)
    np.testing.assert_array_equal(split1.validation_input, split2.validation_input)
    np.testing.assert_array_equal(split1.test_input, split2.test_input)
    np.testing.assert_array_equal(split1.training_output, split2.training_output)
    np.testing.assert_array_equal(split1.validation_output, split2.validation_output)
    np.testing.assert_array_equal(split1.test_output, split2.test_output)

def test_split_sum_of_samples_equals_total() -> None:
    """Test that the sum of training, validation, and test samples equals the total number of samples."""
    x, y = make_data(77, 4)
    split = splitting(x, y, random_state=1, train_data_ratio=0.7, validation_data_ratio=0.2)
    total = (
        split.training_input.shape[0]
        + split.validation_input.shape[0]
        + split.test_input.shape[0]
    )
    assert total == x.shape[0]

def test_invalid_ratios_raise() -> None:
    """Test that invalid train/validation ratios raise ValueError with correct messages."""
    x, y = make_data(10, 2)
    # train_data_ratio=1.0, validation_data_ratio=0.0
    with pytest.raises(ValueError, match="train_data_ratio must be between 0 and 1"):
        splitting(x, y, train_data_ratio=1.0, validation_data_ratio=0.0)
    # train_data_ratio=0.5, validation_data_ratio=0.6 (sum >= 1)
    with pytest.raises(ValueError, match="sum of train_data_ratio and validation_data_ratio must be less than 1"):
        splitting(x, y, train_data_ratio=0.5, validation_data_ratio=0.6)
    # train_data_ratio=-0.1, validation_data_ratio=0.1
    with pytest.raises(ValueError, match="train_data_ratio must be between 0 and 1"):
        splitting(x, y, train_data_ratio=-0.1, validation_data_ratio=0.1)
    # train_data_ratio=0.5, validation_data_ratio='bad'
    with pytest.raises(ValueError, match="train_data_ratio and validation_data_ratio must be convertible to float"):
        splitting(x, y, train_data_ratio=0.5, validation_data_ratio='bad') # type: ignore

def test_non_numpy_inputs_raise() -> None:
    """Test that non-numpy inputs raise ValueError with correct message."""
    x, y = make_data(10, 2)
    with pytest.raises(ValueError, match="Input features and output features must be numpy arrays."):
        splitting([[1, 2], [3, 4]], y) # type: ignore
    with pytest.raises(ValueError, match="Input features and output features must be numpy arrays."):
        splitting(x, [[1, 2], [3, 4]]) # type: ignore

def test_mismatched_lengths_raise() -> None:
    """Test that mismatched lengths of input and output raise ValueError with correct message."""
    x, y = make_data(10, 2)
    y2 = np.arange(9)
    with pytest.raises(ValueError, match="Input features and output features must have the same number of samples"):
        splitting(x, y2)

def test_empty_inputs_raise() -> None:
    """Test that empty inputs raise ValueError with correct message."""
    x = np.empty((0, 2))
    y = np.empty((0,))
    with pytest.raises(ValueError, match="Input features and output features cannot be empty"):
        splitting(x, y)

def test_ratios_as_ints() -> None:
    """Test that ratios can be provided as integers."""
    x, y = make_data(20, 2)
    split = splitting(x, y, train_data_ratio=7/10, validation_data_ratio=2/10)
    n_total = x.shape[0]
    n_train = int(np.floor(0.7 * n_total))
    n_val = int(np.floor(0.2 * n_total))
    n_test = n_total - n_train - n_val
    assert split.training_input.shape[0] == n_train
    assert split.validation_input.shape[0] == n_val
    assert split.test_input.shape[0] == n_test

def test_default_ratios() -> None:
    """Test that default ratios (0.7 for training, 0.15 for validation) work as expected."""
    x, y = make_data(100, 2)
    split = splitting(x, y)
    n_total = x.shape[0]
    n_train = int(np.floor(0.7 * n_total))
    n_val = int(np.floor(0.15 * n_total))
    n_test = n_total - n_train - n_val
    assert split.training_input.shape[0] == n_train
    assert split.validation_input.shape[0] == n_val
    assert split.test_input.shape[0] == n_test

if __name__ == "__main__":
    pytest.main([__file__])
