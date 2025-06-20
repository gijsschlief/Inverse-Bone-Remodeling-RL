"""Test suite for the evaluator module in the bone_remodeling.surrogate_model package."""

import numpy as np
import pytest
import torch
from bone_remodeling.surrogate_model import evaluator


class DummyModel(torch.nn.Module):
    """A simple dummy model for testing purposes."""

    def __init__(self, output_shape=(10, 10)):
        """Initialize the dummy model with a linear layer."""
        super().__init__()
        self.output_shape = output_shape
        self.linear = torch.nn.Linear(3 * 10, 10 * 10)

    def forward(self, x):
        """Forward pass of the dummy model."""
        # Flatten input except batch
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out.view(x.size(0), *self.output_shape)

@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    num_samples = 5
    X_val = np.random.rand(num_samples, 3, 10).astype(np.float32)
    y_val = np.random.rand(num_samples, 10, 10).astype(np.float32)
    return X_val, y_val

@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    return DummyModel()

def test_validate_surrogate_model_shapes(dummy_model, dummy_data):
    """Test that validate_surrogate_model returns predictions and true matrices with correct shapes."""
    X_val, y_val = dummy_data
    # Flatten X_val and y_val to test reshaping inside the function
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    y_val_flat = y_val.reshape(y_val.shape[0], -1)
    preds, trues = evaluator.validate_surrogate_model(
        dummy_model, X_val_flat, y_val_flat, device=torch.device("cpu")
    )
    assert preds.shape == (X_val.shape[0], 10, 10)
    assert trues.shape == (y_val.shape[0], 10, 10)

def test_validate_surrogate_model_wrong_type(dummy_model, dummy_data):
    """Test that validate_surrogate_model raises ValueError for wrong input types."""
    X_val, y_val = dummy_data
    with pytest.raises(ValueError):
        evaluator.validate_surrogate_model(dummy_model, list(X_val), y_val)
    with pytest.raises(ValueError):
        evaluator.validate_surrogate_model(dummy_model, X_val, list(y_val))

def test_validate_surrogate_model_wrong_model(dummy_data):
    """Test that validate_surrogate_model raises ValueError for non-model input."""
    X_val, y_val = dummy_data
    class NotAModel: pass
    with pytest.raises(ValueError):
        evaluator.validate_surrogate_model(NotAModel(), X_val.reshape(X_val.shape[0], -1), y_val.reshape(y_val.shape[0], -1))

def test_validate_surrogate_model_shape_mismatch(dummy_model, dummy_data):
    """Test that validate_surrogate_model raises ValueError for shape mismatch."""
    X_val, y_val = dummy_data
    # X_val wrong shape
    with pytest.raises(ValueError):
        evaluator.validate_surrogate_model(dummy_model, np.random.rand(5, 2, 10), y_val.reshape(5, -1))
    # y_val wrong shape
    with pytest.raises(ValueError):
        evaluator.validate_surrogate_model(dummy_model, X_val.reshape(5, -1), np.random.rand(5, 9, 10))

def test_validate_surrogate_model_sample_mismatch(dummy_model, dummy_data):
    """Test that validate_surrogate_model handles different number of samples in X_val and y_val."""
    X_val, y_val = dummy_data
    # Different number of samples
    X_val2 = X_val[:3].reshape(3, -1)
    y_val2 = y_val[:5].reshape(5, -1)
    preds, trues = evaluator.validate_surrogate_model(dummy_model, X_val2, y_val2)
    assert preds.shape[0] == 3
    assert trues.shape[0] == 3

def test_average_similarity_score(monkeypatch):
    """Test average_similarity_score with a known similarity value."""
    # Patch calculate_similarity to return a known value
    monkeypatch.setattr(evaluator, "calculate_similarity", lambda a, b, **kwargs: 0.5)
    preds = np.zeros((4, 10, 10))
    trues = np.ones((4, 10, 10))
    score = evaluator.average_similarity_score(preds, trues)
    assert score == 0.5

def test_average_similarity_score_partial(monkeypatch):
    """Test average_similarity_score with num_samples parameter."""
    monkeypatch.setattr(evaluator, "calculate_similarity", lambda a, b, **kwargs: 0.8)
    preds = np.zeros((6, 10, 10))
    trues = np.ones((6, 10, 10))
    score = evaluator.average_similarity_score(preds, trues, num_samples=3)
    assert round(score, 6) == 0.8


if __name__ == "__main__":
    pytest.main([__file__])
