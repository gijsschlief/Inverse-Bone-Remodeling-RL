import numpy as np
import warnings


def calculate_similarity(A, B, method="mse", baseline=0.1, threshold=0.5):
    """
    Calculate similarity or distance between two matrices A and B using the specified method.

    Parameters:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.
        method (str): Method to calculate similarity or distance. Options are:
                      "mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein".
        baseline (float): Baseline value for normalization, default is 0.1.
        threshold (float): Threshold for binary operations, default is 0.5.

    Returns:
        float: Calculated similarity or distance score.
    """
    if A is None or B is None:
        raise ValueError("Matrices A and B cannot be None.")
    elif not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("Both A and B must be numpy arrays.")
    elif A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    elif A.size == 0 or B.size == 0:
        raise ValueError("Matrices A and B cannot be empty.")
    elif np.all(A == 0) and np.all(B == 0):
        raise ValueError("Both matrices A and B cannot contain only zeros.")
    elif baseline <= 0:
        raise ValueError("Baseline value must be positive.")
    elif threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if method not in ["mse", "mae", "wasserstein"] and baseline != 0.1:
        warnings.warn(
            f"The 'baseline' parameter is not used for the '{method}' method. Please set it to its default value of 0.1.",
            UserWarning,
        )
    if method not in ["iou", "dice"] and threshold != 0.5:
        warnings.warn(
            f"The 'threshold' parameter is not used for the '{method}' method. Please set it to its default value of 0.5.",
            UserWarning,
        )

    if method == "mse":
        # Mean Squared Error [0, to +inf]
        metric = np.mean((A - B) ** 2)
        return 2 * (1 - metric / (metric + baseline)) - 1  # Normalize to [-1, 1]
    elif method == "mae":
        # Mean Absolute Error [0, to +inf]
        metric = np.mean(np.abs(A - B))
        return 2 * (1 - metric / (metric + baseline)) - 1  # Normalize to [-1, 1]
    elif method == "cosine":
        # Cosine Similarity [-1, 1]
        v1 = A.ravel()
        v2 = B.ravel()
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    elif method == "iou":
        # Intersection over Union (IoU) [0, 1]
        A_bin = (A > threshold).astype(int)
        B_bin = (B > threshold).astype(int)
        intersection = np.logical_and(A_bin, B_bin).sum()
        union = np.logical_or(A_bin, B_bin).sum()
        metric = intersection / (union + 1e-8)
        return 2 * metric - 1  # Normalize to [-1, 1]
    elif method == "dice":
        # Dice Coefficient [0, 1]
        A_bin = (A > threshold).astype(int)
        B_bin = (B > threshold).astype(int)
        intersection = np.logical_and(A_bin, B_bin).sum()
        metric = 2 * intersection / (A_bin.sum() + B_bin.sum() + 1e-8)
        return 2 * metric - 1  # Normalize to [-1, 1]
    elif method == "ssim":
        # Structural Similarity Index (SSIM) [-1, 1]
        from skimage.metrics import structural_similarity as ssim

        score, _ = ssim(A, B, full=True, data_range=A.max() - A.min())
        return score
    elif method == "wasserstein":
        # Earth Mover's Distance (Wasserstein Distance) [0, +inf]
        from scipy.stats import wasserstein_distance

        metric = 0
        for i in range(A.shape[0]):
            metric += wasserstein_distance(A[i, :], B[i, :])
        metric /= A.shape[0]  # Average over rows
        return 2 * (1 - metric / (metric + baseline)) - 1  # Normalize to [-1, 1]
    else:
        raise ValueError(f"Unknown method: {method}")
