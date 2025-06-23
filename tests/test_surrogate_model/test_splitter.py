"""Tests for the splitting function in the splitter module."""

import numpy as np
import pytest

from surrogate_model.splitter import splitting


def test_splitting_shapes_and_sizes():
    """Test that the splitting function returns arrays of correct shapes and sizes."""
    X = np.arange(100).reshape(100, 1)
    Y = np.arange(100)
    X_train, X_val, X_test, y_train, y_val, y_test = splitting(
        X, Y, random_state=42, train_size=0.7, val_size=0.15, test_size=0.15
    )
    assert X_train.shape[0] == 70
    assert X_val.shape[0] == 15
    assert X_test.shape[0] == 15
    assert y_train.shape[0] == 70
    assert y_val.shape[0] == 15
    assert y_test.shape[0] == 15
    # Ensure no overlap
    all_indices = np.concatenate([y_train, y_val, y_test])
    assert set(all_indices) == set(Y)
    assert len(set(all_indices)) == 100


def test_splitting_reproducibility():
    """Test that the splitting function is reproducible with the same random state."""
    X = np.arange(20).reshape(20, 1)
    Y = np.arange(20)
    result1 = splitting(
        X, Y, random_state=123, train_size=0.5, val_size=0.3, test_size=0.2
    )
    result2 = splitting(
        X, Y, random_state=123, train_size=0.5, val_size=0.3, test_size=0.2
    )
    for arr1, arr2 in zip(result1, result2):
        np.testing.assert_array_equal(arr1, arr2)


def test_invalid_train_val_test_sizes_type():
    """Test that the splitting function raises ValueError for invalid train/val/test sizes."""
    X = np.arange(10).reshape(10, 1)
    Y = np.arange(10)
    with pytest.raises(ValueError):
        splitting(X, Y, train_size="0.7", val_size=0.15, test_size=0.15)


def test_invalid_train_val_test_sizes_range():
    """Test that the splitting function raises ValueError for train/val/test sizes out of range."""
    X = np.arange(10).reshape(10, 1)
    Y = np.arange(10)
    with pytest.raises(ValueError):
        splitting(X, Y, train_size=1.0, val_size=0.0, test_size=0.0)


def test_train_val_test_sizes_not_sum_to_one():
    """Test that the splitting function raises AssertionError if train/val/test sizes do not sum to 1."""
    X = np.arange(10).reshape(10, 1)
    Y = np.arange(10)
    with pytest.raises(AssertionError):
        splitting(X, Y, train_size=0.6, val_size=0.2, test_size=0.3)


def test_X_Y_not_numpy_arrays():
    """Test that the splitting function raises TypeError if X or Y are not numpy arrays."""
    X = list(range(10))
    Y = np.arange(10)
    with pytest.raises(ValueError):
        splitting(X, Y, train_size=0.7, val_size=0.15, test_size=0.15)


def test_X_Y_different_lengths():
    """Test that the splitting function raises ValueError if X and Y have different lengths."""
    X = np.arange(10).reshape(10, 1)
    Y = np.arange(9)
    with pytest.raises(ValueError):
        splitting(X, Y, train_size=0.7, val_size=0.15, test_size=0.15)


def test_X_Y_empty():
    """Test that the splitting function raises ValueError if X or Y are empty."""
    X = np.array([]).reshape(0, 1)
    Y = np.array([])
    with pytest.raises(ValueError):
        splitting(X, Y, train_size=0.7, val_size=0.15, test_size=0.15)


if __name__ == "__main__":
    pytest.main([__file__])
