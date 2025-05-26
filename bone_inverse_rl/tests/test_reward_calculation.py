import numpy as np
import pytest
from bone_inverse_rl.rl_model.reward_calculation import calculate_similarity

def test_calculate_similarity_invalid_method():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Unknown method: invalid"):
        calculate_similarity(A, B, method="invalid")

def test_calculate_similarity_type_mismatch():
    A = [[1, 2], [3, 4]]  # Not a numpy array
    B = np.array([[1, 2], [3, 4]])
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        with pytest.raises(TypeError, match="Both A and B must be numpy arrays."):
            calculate_similarity(A, B, method=method)

def test_calculate_similarity_none_input():
    A = None
    B = np.array([[1, 2], [3, 4]])
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        with pytest.raises(ValueError, match="Matrices A and B cannot be None."):
            calculate_similarity(A, B, method=method)

def test_calculate_similarity_negative_baseline():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 5]])
    methods = ["mse", "mae", "wasserstein"]
    for method in methods:
        with pytest.raises(ValueError, match="Baseline value must be positive."):
            calculate_similarity(A, B, method=method, baseline=-1)

def test_calculate_similarity_large_random_matrices():
    A = np.random.rand(1000, 1000)
    B = np.random.rand(1000, 1000)
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        result = calculate_similarity(A, B, method=method)
        assert -1 <= result <= 1, f"Expected result between -1 and 1 for {method}, got {result}"

def test_calculate_similarity_identical_matrices():
    A = np.random.rand(10, 10)
    B = A.copy()
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        result = calculate_similarity(A, B, method=method)
        assert np.isclose(result, 1.0), f"Expected {1.0} for {method}, got {result}"

def test_calculate_similarity_different_shapes():
    A = np.random.rand(10, 10)
    B = np.random.rand(5, 5)
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        with pytest.raises(ValueError, match="Matrices A and B must have the same shape."):
            calculate_similarity(A, B, method=method)

def test_calculate_similarity_all_zeros():
    A = np.zeros((10, 10))
    B = np.zeros((10, 10))
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        with pytest.raises(ValueError, match="Both matrices A and B cannot contain only zeros."):
            calculate_similarity(A, B, method=method)

def test_calculate_similarity_random_matrices_with_baseline():
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    baseline = 2.0
    methods = ["mse", "mae", "wasserstein"]
    for method in methods:
        result = calculate_similarity(A, B, method=method, baseline=baseline)
        assert -1 <= result <= 1, f"Expected result between -1 and 1 for {method} with baseline, got {result}"

def test_calculate_similarity_varied_size_matrices():
    sizes = [(7, 7), (10, 10), (15, 15), (50, 50), (100, 100)]
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for size in sizes:
        A = np.random.rand(*size)
        B = np.random.rand(*size)
        for method in methods:
            result = calculate_similarity(A, B, method=method)
            assert -1 <= result <= 1, f"Expected result between -1 and 1 for {method} with size {size}, got {result}"

def test_calculate_similarity_empty_matrices():
    A = np.array([])
    B = np.array([])
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        with pytest.raises(ValueError, match="Matrices A and B cannot be empty."):
            calculate_similarity(A, B, method=method)

def test_calculate_similarity_random_matrices():
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        for _ in range(1000):  # Test with 100 random pairs of matrices
            A = np.abs(np.random.rand(10, 10))
            B = np.abs(np.random.rand(10, 10))
            result = calculate_similarity(A, B, method=method)
            assert -1 <= result <= 1, f"Expected SSIM between -1 and 1 for {method}, got {result}"

def test_calculate_similarity_similar_but_not_equal_matrices():
    A = np.random.rand(10, 10)
    B = A + np.random.normal(0, 0.1, (10, 10))  # Adding small noise to make B slightly different from A
    methods = ["mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein"]
    for method in methods:
        result = calculate_similarity(A, B, method=method)
        assert result > 0, f"Expected positive result for similar matrices with {method}, got {result}"

def test_calculate_similarity_thresholding():
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    threshold = 0.5
    methods = ["iou", "dice"]
    for method in methods:
        result = calculate_similarity(A, B, method=method, threshold=threshold)
        assert -1 <= result <= 1, f"Expected result between -1 and 1 for {method} with threshold, got {result}"

def test_calculate_similarity_unused_baseline_warning():
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    methods = ["cosine", "iou", "dice", "ssim"]
    for method in methods:
        with pytest.warns(UserWarning, match=f"The 'baseline' parameter is not used for the '{method}' method."):
            calculate_similarity(A, B, method=method, baseline=2.0)

def test_calculate_similarity_unused_threshold_warning():
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    methods = ["mse", "mae", "cosine", "ssim", "wasserstein"]
    for method in methods:
        with pytest.warns(UserWarning, match=f"The 'threshold' parameter is not used for the '{method}' method."):
            calculate_similarity(A, B, method=method, threshold=0.7)

def test_calculate_similarity_dissimilar_matrices():
    A = np.ones((10, 10))
    B = np.random.rand(10, 10) * 0.1
    methods = ["mse", "mae", "iou", "dice", "wasserstein"]
    for method in methods:
        result = calculate_similarity(A, B, method=method)
        assert result < 0, f"Expected negative result for dissimilar matrices with {method}, got {result}"
    # Special case for SSIM
    result_ssim = calculate_similarity(A, B, method="ssim")
    assert result_ssim <= 0, f"Expected non-positive result for dissimilar matrices with ssim, got {result_ssim}"

def test_calculate_similarity_warning_and_result():
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    with pytest.warns(UserWarning, match="The 'baseline' parameter is not used for the 'cosine' method."):
        result = calculate_similarity(A, B, method="cosine", baseline=2.0)
    assert -1 <= result <= 1, f"Expected result between -1 and 1 for cosine method, got {result}"

def test_calculate_similarity_cosine_orthogonal():
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[0, 1], [-1, 0]])  # Orthogonal to A
    result = calculate_similarity(A, B, method="cosine")
    assert np.isclose(result, 0.0), f"Expected 0 for cosine similarity of orthogonal matrices, got {result}"

def test_calculate_similarity_cosine_opposite():
    A = np.array([[1, 0], [0, 1]])
    B = -A  # Opposite to A
    result = calculate_similarity(A, B, method="cosine")
    assert np.isclose(result, -1.0), f"Expected -1 for cosine similarity of opposite matrices, got {result}"