"""Test cases for the visualizer module in the surrogate model package."""

import matplotlib
import numpy as np
import pytest

from bone_remodeling.src.surrogate_model.visualizer import plot_surrogate_model

matplotlib.use("Agg")  # Use non-interactive backend for testing


def test_plot_surrogate_model_returns_figures() -> None:
    """Test that plot_surrogate_model returns a list of matplotlib figures."""
    # Arrange
    predicted_matrices = np.ones((4, 5, 5))
    true_matrices = np.ones((4, 5, 5)) * 2
    sample_count = 2

    # Act
    figures = plot_surrogate_model(
        true_matrices,
        predicted_matrices,
        sample_count=sample_count,
        show_plot=False,
    )

    # Assert
    assert isinstance(figures, list)
    assert len(figures) == sample_count
    for fig in figures:
        assert hasattr(fig, "savefig")  # matplotlib Figure


def test_plot_surrogate_model_invalid_array_type() -> None:
    """Test that plot_surrogate_model raises ValueError for invalid array type."""
    # Arrange
    true_matrices = np.ones((3, 4, 4))
    predicted_matrices = "not an array"

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="Both predicted_matrices and true_matrices must be numpy.ndarray objects.",
    ):
        plot_surrogate_model(true_matrices, predicted_matrices, show_plot=False)  # type: ignore


def test_plot_surrogate_model_shape_mismatch() -> None:
    """Test that plot_surrogate_model raises ValueError for shape mismatch."""
    true_matrices = np.ones((3, 4, 4))
    predicted_matrices = np.ones((2, 4, 4))
    with pytest.raises(ValueError, match="must have the same shape"):
        plot_surrogate_model(true_matrices, predicted_matrices, show_plot=False)


def test_plot_surrogate_model_invalid_sample_count() -> None:
    """Test that plot_surrogate_model raises ValueError for invalid sample_count."""
    true_matrices = np.ones((3, 4, 4))
    predicted_matrices = np.ones((3, 4, 4))
    with pytest.raises(ValueError, match="sample_count must be a positive integer"):
        plot_surrogate_model(
            true_matrices,
            predicted_matrices,
            sample_count=0,
            show_plot=False,
        )
    with pytest.raises(ValueError, match="sample_count must be a positive integer"):
        plot_surrogate_model(
            true_matrices,
            predicted_matrices,
            sample_count=4,
            show_plot=False,
        )


if __name__ == "__main__":
    pytest.main([__file__])
