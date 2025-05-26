import numpy as np
import pytest
from bone_inverse_rl.rl_model.reward_calculation import calculate_similarity, ssim, wasserstein_distance

def test_calculate_similarity_mse():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 5]])
    result = calculate_similarity(A, B, method="mse")
    expected = 0.25  # Mean Squared Error
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_mae():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 5]])
    result = calculate_similarity(A, B, method="mae")
    expected = 0.5  # Mean Absolute Error
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_cosine():
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[0, 1], [1, 0]])
    result = calculate_similarity(A, B, method="cosine")
    expected = 0.0  # Cosine similarity for orthogonal vectors
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_iou():
    A = np.array([[0.6, 0.4], [0.8, 0.2]])
    B = np.array([[0.7, 0.3], [0.9, 0.1]])
    result = calculate_similarity(A, B, method="iou")
    expected = 1.0  # Perfect overlap after thresholding
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_dice():
    A = np.array([[0.6, 0.4], [0.8, 0.2]])
    B = np.array([[0.7, 0.3], [0.9, 0.1]])
    result = calculate_similarity(A, B, method="dice")
    expected = 1.0  # Perfect overlap after thresholding
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_ssim():
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    B = np.array([[1, 2], [3, 4]], dtype=np.float64)
    result = calculate_similarity(A, B, method="ssim")
    expected = 1.0  # Identical matrices
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_wasserstein():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 5]])
    result = calculate_similarity(A, B, method="wasserstein")
    expected = 0.25  # Wasserstein distance for flattened arrays
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_invalid_method():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Unknown method: invalid"):
        calculate_similarity(A, B, method="invalid")

def test_calculate_similarity_shape_mismatch():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2]])
    with pytest.raises(ValueError, match="Matrices A and B must have the same shape."):
        calculate_similarity(A, B)

def test_calculate_similarity_mse_large_matrices():
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)
    result = calculate_similarity(A, B, method="mse")
    assert result >= 0, f"Expected non-negative MSE, got {result}"

def test_calculate_similarity_mae_large_matrices():
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)
    result = calculate_similarity(A, B, method="mae")
    assert result >= 0, f"Expected non-negative MAE, got {result}"

def test_calculate_similarity_cosine_identical():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 4]])
    result = calculate_similarity(A, B, method="cosine")
    expected = 1.0  # Cosine similarity for identical matrices
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_iou_no_overlap():
    A = np.array([[0.6, 0.4], [0.8, 0.2]])
    B = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = calculate_similarity(A, B, method="iou")
    expected = 0.0  # No overlap after thresholding
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_dice_no_overlap():
    A = np.array([[0.6, 0.4], [0.8, 0.2]])
    B = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = calculate_similarity(A, B, method="dice")
    expected = 0.0  # No overlap after thresholding
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_ssim_different_matrices():
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    B = np.array([[4, 3], [2, 1]], dtype=np.float64)
    result = calculate_similarity(A, B, method="ssim")
    assert 0 <= result < 1, f"Expected SSIM between 0 and 1 for different matrices, got {result}"

def test_calculate_similarity_wasserstein_identical():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 4]])
    result = calculate_similarity(A, B, method="wasserstein")
    expected = 0.0  # Wasserstein distance for identical matrices
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_random_method():
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    with pytest.raises(ValueError, match="Unknown method: random"):
        calculate_similarity(A, B, method="random")

def test_calculate_similarity_mse_large_values():
    A = np.array([[1e10, 2e10], [3e10, 4e10]])
    B = np.array([[1e10, 2e10], [3e10, 5e10]])
    result = calculate_similarity(A, B, method="mse")
    expected = 2.5e19  # Mean Squared Error for large values
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_mae_large_values():
    A = np.array([[1e10, 2e10], [3e10, 4e10]])
    B = np.array([[1e10, 2e10], [3e10, 5e10]])
    result = calculate_similarity(A, B, method="mae")
    expected = 5e9  # Mean Absolute Error for large values
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_cosine_negative_values():
    A = np.array([[-1, -2], [-3, -4]])
    B = np.array([[-1, -2], [-3, -4]])
    result = calculate_similarity(A, B, method="cosine")
    expected = 1.0  # Cosine similarity for identical negative matrices
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_iou_all_zeros():
    A = np.zeros((2, 2))
    B = np.zeros((2, 2))
    result = calculate_similarity(A, B, method="iou")
    expected = 0.0  # IoU for matrices with no positive values
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_dice_all_zeros():
    A = np.zeros((2, 2))
    B = np.zeros((2, 2))
    result = calculate_similarity(A, B, method="dice")
    expected = 0.0  # Dice coefficient for matrices with no positive values
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_ssim_large_matrices():
    A = np.random.rand(1000, 1000)
    B = np.random.rand(1000, 1000)
    result = calculate_similarity(A, B, method="ssim")
    assert 0 <= result <= 1, f"Expected SSIM between 0 and 1, got {result}"

def test_calculate_similarity_wasserstein_large_values():
    A = np.array([[1e10, 2e10], [3e10, 4e10]])
    B = np.array([[1e10, 2e10], [3e10, 5e10]])
    result = calculate_similarity(A, B, method="wasserstein")
    expected = 2.5e9  # Wasserstein distance for large values
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_mse_empty_matrices():
    A = np.array([])
    B = np.array([])
    with pytest.raises(ValueError):
        calculate_similarity(A, B, method="mse")

def test_calculate_similarity_mae_empty_matrices():
    A = np.array([])
    B = np.array([])
    with pytest.raises(ValueError):
        calculate_similarity(A, B, method="mae")

def test_calculate_similarity_cosine_empty_matrices():
    A = np.array([])
    B = np.array([])
    with pytest.raises(ValueError):
        calculate_similarity(A, B, method="cosine")

def test_calculate_similarity_iou_empty_matrices():
    A = np.array([])
    B = np.array([])
    with pytest.raises(ValueError):
        calculate_similarity(A, B, method="iou")

def test_calculate_similarity_dice_empty_matrices():
    A = np.array([])
    B = np.array([])
    with pytest.raises(ValueError):
        calculate_similarity(A, B, method="dice")

def test_calculate_similarity_ssim_empty_matrices():
    A = np.array([])
    B = np.array([])
    with pytest.raises(ValueError):
        calculate_similarity(A, B, method="ssim")

def test_calculate_similarity_wasserstein_empty_matrices():
    A = np.array([])
    B = np.array([])
    with pytest.raises(ValueError):
        calculate_similarity(A, B, method="wasserstein")

def test_calculate_similarity_iou_partial_overlap():
    A = np.array([[0.6, 0.4], [0.8, 0.2]])
    B = np.array([[0.6, 0.4], [0.2, 0.8]])
    result = calculate_similarity(A, B, method="iou")
    expected = 0.5  # Partial overlap after thresholding
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

def test_calculate_similarity_dice_partial_overlap():
    A = np.array([[0.6, 0.4], [0.8, 0.2]])
    B = np.array([[0.6, 0.4], [0.2, 0.8]])
    result = calculate_similarity(A, B, method="dice")
    expected = 0.6666666666666666  # Partial overlap after thresholding
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"
