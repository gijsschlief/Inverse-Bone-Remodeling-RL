"""Test cases for the reward calculation module in the RL model."""

import numpy as np
import pytest

from bone_remodeling.src.rl_model.reward_calculation import calculate_similarity


def test_calculate_similarity_invalid_method() -> None:
    """Test that an invalid method raises a ValueError."""
    first_matrix = np.array([[1, 2], [3, 4]])
    second_matrix = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Unknown method: invalid"):
        calculate_similarity(first_matrix, second_matrix, method="invalid")


def test_calculate_similarity_type_mismatch() -> None:
    """Test that a TypeError is raised when inputs are not numpy arrays."""
    first_matrix = [[1, 2], [3, 4]]  # Not a numpy array
    second_matrix = np.array([[1, 2], [3, 4]])
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        with pytest.raises(TypeError, match="Both A and B must be numpy arrays."):
            calculate_similarity(first_matrix, second_matrix, method=method)  # type: ignore


def test_calculate_similarity_none_input() -> None:
    """Test that a ValueError is raised when either input is None."""
    first_matrix = None
    second_matrix = np.array([[1, 2], [3, 4]])
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        with pytest.raises(ValueError, match="Matrices A and B cannot be None."):
            calculate_similarity(first_matrix, second_matrix, method=method)  # type: ignore


def test_calculate_similarity_negative_baseline() -> None:
    """Test that a ValueError is raised when baseline is negative."""
    first_matrix = np.array([[1, 2], [3, 4]])
    second_matrix = np.array([[1, 2], [3, 5]])
    methods = ["mse", "mae", "wasserstein"]
    for method in methods:
        with pytest.raises(ValueError, match="Baseline value must be positive."):
            calculate_similarity(
                first_matrix, second_matrix, method=method, baseline=-1
            )


def test_calculate_similarity_large_random_matrices() -> None:
    """Test that similarity calculation works for large random matrices."""
    first_matrix = np.random.rand(1000, 1000)
    second_matrix = np.random.rand(1000, 1000)
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        result = calculate_similarity(first_matrix, second_matrix, method=method)
        assert (
            -1 <= result <= 1
        ), f"Expected result between -1 and 1 for {method}, got {result}"


def test_calculate_similarity_identical_matrices() -> None:
    """Test that similarity calculation returns 1 for identical matrices."""
    first_matrix = np.random.rand(10, 10)
    second_matrix = first_matrix.copy()
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        result = calculate_similarity(first_matrix, second_matrix, method=method)
        assert np.isclose(result, 1.0), f"Expected {1.0} for {method}, got {result}"


def test_calculate_similarity_different_shapes() -> None:
    """Test that a ValueError is raised for matrices of different shapes."""
    first_matrix = np.random.rand(10, 10)
    second_matrix = np.random.rand(5, 5)
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        with pytest.raises(
            ValueError, match="Matrices A and B must have the same shape."
        ):
            calculate_similarity(first_matrix, second_matrix, method=method)


def test_calculate_similarity_all_zeros() -> None:
    """Test that a ValueError is raised when both matrices are all zeros."""
    first_matrix = np.zeros((10, 10))
    second_matrix = np.zeros((10, 10))
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        with pytest.raises(
            ValueError, match="Both matrices A and B cannot contain only zeros."
        ):
            calculate_similarity(first_matrix, second_matrix, method=method)


def test_calculate_similarity_random_matrices_with_baseline() -> None:
    """Test that similarity calculation works with a baseline for random matrices."""
    first_matrix = np.random.rand(10, 10)
    second_matrix = np.random.rand(10, 10)
    baseline = 2.0
    methods = ["mse", "mae", "wasserstein"]
    for method in methods:
        result = calculate_similarity(
            first_matrix, second_matrix, method=method, baseline=baseline
        )
        assert (
            -1 <= result <= 1
        ), f"Expected result between -1 and 1 for {method} with baseline, got {result}"


def test_calculate_similarity_varied_size_matrices() -> None:
    """Test that similarity calculation works for matrices of varied sizes."""
    sizes = [(7, 7), (10, 10), (15, 15), (50, 50), (100, 100)]
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for size in sizes:
        first_matrix = np.random.rand(*size)
        second_matrix = np.random.rand(*size)
        for method in methods:
            result = calculate_similarity(first_matrix, second_matrix, method=method)
            assert (
                -1 <= result <= 1
            ), f"Expected result between -1 and 1 for {method} with size {size}, got {result}"


def test_calculate_similarity_empty_matrices() -> None:
    """Test that a ValueError is raised when either matrix is empty."""
    first_matrix = np.array([])
    second_matrix = np.array([])
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        with pytest.raises(ValueError, match="Matrices A and B cannot be empty."):
            calculate_similarity(first_matrix, second_matrix, method=method)


def test_calculate_similarity_random_matrices() -> None:
    """Test that similarity calculation works for random matrices."""
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        for _ in range(1000):  # Test with 100 random pairs of matrices
            first_matrix = np.abs(np.random.rand(10, 10))
            second_matrix = np.abs(np.random.rand(10, 10))
            result = calculate_similarity(first_matrix, second_matrix, method=method)
            assert (
                -1 <= result <= 1
            ), f"Expected SSIM between -1 and 1 for {method}, got {result}"


def test_calculate_similarity_similar_but_not_equal_matrices() -> None:
    """Test that similarity calculation works for matrices that are similar but not equal."""
    first_matrix = np.random.rand(10, 10)
    second_matrix = first_matrix + np.random.normal(
        0, 0.1, (10, 10)
    )  # Adding small noise to make B slightly different from A
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        result = calculate_similarity(first_matrix, second_matrix, method=method)
        assert (
            result > 0
        ), f"Expected positive result for similar matrices with {method}, got {result}"


def test_calculate_similarity_thresholding() -> None:
    """Test that similarity calculation works with thresholding."""
    first_matrix = np.random.rand(10, 10)
    second_matrix = np.random.rand(10, 10)
    threshold = 0.5
    methods = ["iou", "dice"]
    for method in methods:
        result = calculate_similarity(
            first_matrix, second_matrix, method=method, threshold=threshold
        )
        assert (
            -1 <= result <= 1
        ), f"Expected result between -1 and 1 for {method} with threshold, got {result}"


def test_calculate_similarity_unused_baseline_warning() -> None:
    """Test that a warning is raised when baseline is unused for certain methods."""
    first_matrix = np.random.rand(10, 10)
    second_matrix = np.random.rand(10, 10)
    methods = ["cosine", "iou", "dice", "ssim"]
    for method in methods:
        with pytest.warns(
            UserWarning,
            match=f"The 'baseline' parameter is not used for the '{method}' method.",
        ):
            calculate_similarity(
                first_matrix, second_matrix, method=method, baseline=2.0
            )


def test_calculate_similarity_unused_threshold_warning() -> None:
    """Test that a warning is raised when threshold is unused for certain methods."""
    first_matrix = np.random.rand(10, 10)
    second_matrix = np.random.rand(10, 10)
    methods = ["mse", "mae", "cosine", "ssim", "wasserstein"]
    for method in methods:
        with pytest.warns(
            UserWarning,
            match=f"The 'threshold' parameter is not used for the '{method}' method.",
        ):
            calculate_similarity(
                first_matrix, second_matrix, method=method, threshold=0.7
            )


def test_calculate_similarity_dissimilar_matrices() -> None:
    """Test that similarity calculation returns negative values for dissimilar matrices."""
    first_matrix = np.ones((10, 10))
    second_matrix = np.random.rand(10, 10) * 0.1
    methods = ["mse", "mae", "iou", "dice", "wasserstein"]
    for method in methods:
        result = calculate_similarity(first_matrix, second_matrix, method=method)
        assert (
            result < 0
        ), f"Expected negative result for dissimilar matrices with {method}, got {result}"
    # Special case for SSIM
    result_ssim = calculate_similarity(first_matrix, second_matrix, method="ssim")
    assert (
        result_ssim <= 0
    ), f"Expected non-positive result for dissimilar matrices with ssim, got {result_ssim}"


def test_calculate_similarity_warning_and_result() -> None:
    """Test that a warning is raised and result is valid for cosine method with baseline."""
    first_matrix = np.random.rand(10, 10)
    second_matrix = np.random.rand(10, 10)
    with pytest.warns(
        UserWarning,
        match="The 'baseline' parameter is not used for the 'cosine' method.",
    ):
        result = calculate_similarity(
            first_matrix, second_matrix, method="cosine", baseline=2.0
        )
    assert (
        -1 <= result <= 1
    ), f"Expected result between -1 and 1 for cosine method, got {result}"


def test_calculate_similarity_cosine_orthogonal() -> None:
    """Test that cosine similarity returns 0 for orthogonal matrices."""
    first_matrix = np.array([[1, 0], [0, 1]])
    second_matrix = np.array([[0, 1], [-1, 0]])  # Orthogonal to A
    result = calculate_similarity(first_matrix, second_matrix, method="cosine")
    assert np.isclose(
        result, 0.0
    ), f"Expected 0 for cosine similarity of orthogonal matrices, got {result}"


def test_calculate_similarity_cosine_opposite() -> None:
    """Test that cosine similarity returns -1 for opposite matrices."""
    first_matrix = np.array([[1, 0], [0, 1]])
    second_matrix = -first_matrix  # Opposite to A
    result = calculate_similarity(first_matrix, second_matrix, method="cosine")
    assert np.isclose(
        result, -1.0
    ), f"Expected -1 for cosine similarity of opposite matrices, got {result}"


if __name__ == "__main__":
    pytest.main([__file__])
