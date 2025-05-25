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