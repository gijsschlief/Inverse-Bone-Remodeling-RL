"""Test cases for the visualizer module in the surrogate model package."""

import matplotlib
import numpy as np
import pytest
from bone_remodeling.surrogate_model import visualizer

matplotlib.use("Agg")  # Use non-interactive backend for testing


def test_plot_surrogate_model_returns_figures(monkeypatch):
    """Test that plot_surrogate_model returns a list of matplotlib figures."""
    # Arrange
    predicted_matrices = np.ones((4, 5, 5))
    true_matrices = np.ones((4, 5, 5)) * 2
    sample_count = 2

    # Patch plt.show to avoid opening windows during tests
    monkeypatch.setattr(visualizer.plt, "show", lambda: None)

    # Act
    figures = visualizer.plot_surrogate_model(
        true_matrices, predicted_matrices, sample_count=sample_count, show_plot=True
    )

    # Assert
    assert isinstance(figures, list)
    assert len(figures) == sample_count
    for fig in figures:
        assert hasattr(fig, "savefig")  # matplotlib Figure


def test_plot_surrogate_model_invalid_array_type():
    """Test that plot_surrogate_model raises ValueError for invalid array type."""
    # Arrange
    true_matrices = np.ones((3, 4, 4))
    predicted_matrices = "not an array"

    # Act & Assert
    with pytest.raises(ValueError):
        visualizer.plot_surrogate_model(true_matrices, predicted_matrices)


def test_plot_surrogate_model_shape_mismatch():
    """Test that plot_surrogate_model raises ValueError for shape mismatch."""
    true_matrices = np.ones((3, 4, 4))
    predicted_matrices = np.ones((2, 4, 4))
    with pytest.raises(ValueError, match="must have the same shape"):
        visualizer.plot_surrogate_model(true_matrices, predicted_matrices)


def test_plot_surrogate_model_invalid_sample_count():
    """Test that plot_surrogate_model raises ValueError for invalid sample_count."""
    true_matrices = np.ones((3, 4, 4))
    predicted_matrices = np.ones((3, 4, 4))
    with pytest.raises(ValueError, match="sample_count must be a positive integer"):
        visualizer.plot_surrogate_model(
            true_matrices, predicted_matrices, sample_count=0
        )
    with pytest.raises(ValueError, match="sample_count must be a positive integer"):
        visualizer.plot_surrogate_model(
            true_matrices, predicted_matrices, sample_count=4
        )


def test_plot_density_matrix_creates_annotations():
    """Test that plot_density_matrix creates text annotations for matrix values."""
    import matplotlib.pyplot as plt

    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    fig, ax = plt.subplots()
    visualizer.plot_density_matrix(matrix, "Test Matrix", ax, vmin=1.0, vmax=4.0)
    # Check that the axis has the correct title and labels
    assert ax.get_title() == "Test Matrix"
    assert ax.get_xlabel() == "Columns"
    assert ax.get_ylabel() == "Rows"
    # Check that text annotations exist
    texts = [
        child for child in ax.get_children() if isinstance(child, matplotlib.text.Text)
    ]
    assert any("1.00" in t.get_text() for t in texts)
    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
