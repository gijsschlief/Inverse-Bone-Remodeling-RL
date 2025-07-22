"""Test suite for the evaluator module in the bone_remodeling.surrogate_model package."""

import re
from typing import Tuple

import numpy as np
import pytest
import torch

from bone_remodeling.src.surrogate_model import evaluator


class DummyModel(torch.nn.Module):
    """A simple dummy model for testing purposes."""

    def __init__(self, output_shape: Tuple[int, int] = (10, 10)) -> None:
        """Initialize the dummy model with a linear layer."""
        super().__init__()
        self.output_shape = output_shape
        self.linear = torch.nn.Linear(3 * 10, 10 * 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the dummy model."""
        # Flatten input except batch
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out.view(x.size(0), *self.output_shape)


@pytest.fixture
def dummy_data() -> Tuple[np.ndarray, np.ndarray]:
    """Create dummy data for testing."""
    num_samples = 5
    x_val = np.random.rand(num_samples, 3, 10).astype(np.float32)
    y_val = np.random.rand(num_samples, 10, 10).astype(np.float32)
    return x_val, y_val


@pytest.fixture
def dummy_model() -> DummyModel:
    """Create a dummy model for testing."""
    return DummyModel()


def test_validate_surrogate_model_shapes(
    dummy_model: DummyModel, dummy_data: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test that validate_surrogate_model returns predictions and true matrices with correct shapes."""
    x_val, y_val = dummy_data
    # Flatten x_val and y_val to test reshaping inside the function
    x_val_flat = x_val.reshape(x_val.shape[0], -1)
    y_val_flat = y_val.reshape(y_val.shape[0], -1)
    preds, trues = evaluator.validate_surrogate_model(
        dummy_model, x_val_flat, y_val_flat, device=torch.device("cpu")
    )
    assert preds.shape == (x_val.shape[0], 10, 10)
    assert trues.shape == (y_val.shape[0], 10, 10)


def test_validate_surrogate_model_wrong_type(
    dummy_model: DummyModel, dummy_data: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test that validate_surrogate_model raises ValueError for wrong input types."""
    x_val, y_val = dummy_data
    with pytest.raises(
        ValueError, match="Both X_val and y_val must be numpy.ndarray objects."
    ):
        evaluator.validate_surrogate_model(dummy_model, list(x_val), y_val)  # type: ignore
    with pytest.raises(
        ValueError, match="Both X_val and y_val must be numpy.ndarray objects."
    ):
        evaluator.validate_surrogate_model(dummy_model, x_val, list(y_val))  # type: ignore


def test_validate_surrogate_model_wrong_model(
    dummy_data: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test that validate_surrogate_model raises ValueError for non-model input."""
    x_val, y_val = dummy_data

    class NotAModel:
        pass

    with pytest.raises(
        ValueError, match="The model must be an instance of torch.nn.Module."
    ):
        evaluator.validate_surrogate_model(
            NotAModel(),  # type: ignore
            x_val.reshape(x_val.shape[0], -1),
            y_val.reshape(y_val.shape[0], -1),
        )


def test_validate_surrogate_model_shape_mismatch(
    dummy_model: DummyModel, dummy_data: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test that validate_surrogate_model raises ValueError for shape mismatch."""
    x_val, y_val = dummy_data
    num_samples = np.min([x_val.shape[0], y_val.shape[0]])
    # x_val wrong shape
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"x_val with shape (5, 2, 10) cannot be reshaped to ({num_samples}, 3, 10)."
        ),
    ):
        evaluator.validate_surrogate_model(
            dummy_model, np.random.rand(5, 2, 10), y_val.reshape(5, -1)
        )
    # y_val wrong shape
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"y_val with shape (5, 9, 10) cannot be reshaped to ({num_samples}, 10, 10)."
        ),
    ):
        evaluator.validate_surrogate_model(
            dummy_model, x_val.reshape(5, -1), np.random.rand(5, 9, 10)
        )


def test_validate_surrogate_model_sample_mismatch(
    dummy_model: DummyModel, dummy_data: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test that validate_surrogate_model handles different number of samples in x_val and y_val."""
    x_val, y_val = dummy_data
    # Different number of samples
    x_val2 = x_val[:3].reshape(3, -1)
    y_val2 = y_val[:5].reshape(5, -1)
    preds, trues = evaluator.validate_surrogate_model(dummy_model, x_val2, y_val2)
    assert preds.shape[0] == 3
    assert trues.shape[0] == 3


def test_average_similarity_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test average_similarity_score with a known similarity value."""
    # Patch calculate_similarity to return a known value
    monkeypatch.setattr(evaluator, "calculate_similarity", lambda a, b, **kwargs: 0.5)
    preds = np.zeros((4, 10, 10))
    trues = np.ones((4, 10, 10))
    score = evaluator.average_similarity_score(preds, trues)
    assert score == 0.5


def test_average_similarity_score_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test average_similarity_score with num_samples parameter."""
    monkeypatch.setattr(evaluator, "calculate_similarity", lambda a, b, **kwargs: 0.8)
    preds = np.zeros((6, 10, 10))
    trues = np.ones((6, 10, 10))
    score = evaluator.average_similarity_score(preds, trues, num_samples=3)
    assert round(score, 6) == 0.8


if __name__ == "__main__":
    pytest.main([__file__])
