"""Calculate similarity or distance between two matrices using various methods."""

import warnings

import numpy as np


def calculate_similarity(
    reference_matrix: np.ndarray,
    comparison_matrix: np.ndarray,
    method: str = "mse",
    baseline: float = 0.1,
    threshold: float = 0.5,
) -> float:
    """Calculate similarity or distance between two matrices A and B using the specified method.

    Args:
    ----
        reference_matrix (np.ndarray): First matrix for comparison.
        comparison_matrix (np.ndarray): Second matrix for comparison.
        method (str): Method to calculate similarity or distance. Options are:
                      "mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein".
        baseline (float): Baseline value for normalization, default is 0.1.
        threshold (float): Threshold for binary operations, default is 0.5.

    Returns:
    -------
        float: Calculated similarity or distance score.

    """
    if reference_matrix is None or comparison_matrix is None:
        raise ValueError("Matrices A and B cannot be None.")
    elif not isinstance(reference_matrix, np.ndarray) or not isinstance(comparison_matrix, np.ndarray):
        raise TypeError("Both A and B must be numpy arrays.")
    elif reference_matrix.shape != comparison_matrix.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    elif reference_matrix.size == 0 or comparison_matrix.size == 0:
        raise ValueError("Matrices A and B cannot be empty.")
    elif np.all(reference_matrix == 0) and np.all(comparison_matrix == 0):
        raise ValueError("Both matrices A and B cannot contain only zeros.")
    elif baseline <= 0:
        raise ValueError("Baseline value must be positive.")
    elif threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if method not in ["mse", "mae", "wasserstein"] and baseline != 0.1:
        warnings.warn(
            f"The 'baseline' parameter is not used for the '{method}' method. Please set it to its default value of 0.1.",
            UserWarning,
            stacklevel=2,
        )
    if method not in ["iou", "dice"] and threshold != 0.5:
        warnings.warn(
            f"The 'threshold' parameter is not used for the '{method}' method. Please set it to its default value of 0.5.",
            UserWarning,
            stacklevel=2,
        )

    if method == "mse":
        # Mean Squared Error [0, to +inf]
        metric = np.mean((reference_matrix - comparison_matrix) ** 2)
        return 2 * (1 - metric / (metric + baseline)) - 1  # Normalize to [-1, 1]

    elif method == "mae":
        # Mean Absolute Error [0, to +inf]
        metric = np.mean(np.abs(reference_matrix - comparison_matrix))
        return 2 * (1 - metric / (metric + baseline)) - 1  # Normalize to [-1, 1]

    elif method == "cosine":
        # Cosine Similarity [-1, 1]
        v1 = reference_matrix.ravel()
        v2 = comparison_matrix.ravel()
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    elif method == "iou":
        # Intersection over Union (IoU) [0, 1]
        a_thresholded_array = (reference_matrix > threshold).astype(int)
        b_thresholded_array = (comparison_matrix > threshold).astype(int)
        intersection = np.logical_and(a_thresholded_array, b_thresholded_array).sum()
        union = np.logical_or(a_thresholded_array, b_thresholded_array).sum()
        metric = intersection / (union + 1e-8)
        return 2 * metric - 1  # Normalize to [-1, 1]

    elif method == "dice":
        # Dice Coefficient [0, 1]
        a_thresholded_array = (reference_matrix > threshold).astype(int)
        b_thresholded_array = (comparison_matrix > threshold).astype(int)
        intersection = np.logical_and(a_thresholded_array, b_thresholded_array).sum()
        metric = 2 * intersection / (a_thresholded_array.sum() + b_thresholded_array.sum() + 1e-8)
        return 2 * metric - 1  # Normalize to [-1, 1]

    elif method == "ssim":
        # Structural Similarity Index (SSIM) [-1, 1]
        from skimage.metrics import structural_similarity as ssim

        score, _ = ssim(reference_matrix, comparison_matrix, full=True, data_range=reference_matrix.max() - reference_matrix.min())
        return score

    elif method == "wasserstein":
        # Earth Mover's Distance (Wasserstein Distance) [0, +inf]
        from scipy.stats import wasserstein_distance  # type: ignore

        metric = 0
        for i in range(reference_matrix.shape[0]):
            metric += wasserstein_distance(reference_matrix[i, :], comparison_matrix[i, :])
        metric /= reference_matrix.shape[0]  # Average over rows
        return 2 * (1 - metric / (metric + baseline)) - 1  # Normalize to [-1, 1]

    else:
        raise ValueError(f"Unknown method: {method}")
