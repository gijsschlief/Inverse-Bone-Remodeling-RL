import numpy as np

def calculate_similarity(A, B, method="mse"):
    """
    Calculate similarity or distance between two matrices A and B using the specified method.

    Parameters:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.
        method (str): Method to calculate similarity or distance. Options are:
                      "mse", "mae", "cosine", "iou", "dice", "ssim", "wasserstein".

    Returns:
        float: Calculated similarity or distance score.
    """
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")
    elif not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("Both A and B must be numpy arrays.")
    elif A is None or B is None:
        raise ValueError("Matrices A and B cannot be None.")
    elif A .size == 0 or B.size == 0:
        raise ValueError("Matrices A and B cannot be empty.")

    if method == "mse":
        # Mean Squared Error
        return np.mean((A - B) ** 2)
    elif method == "mae":
        # Mean Absolute Error
        return np.mean(np.abs(A - B))
    elif method == "cosine":
        # Cosine Similarity
        v1 = A.ravel()
        v2 = B.ravel()
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    elif method == "iou":
        # Intersection over Union (IoU)
        thr = 0.5
        A_bin = (A > thr).astype(int)
        B_bin = (B > thr).astype(int)
        intersection = np.logical_and(A_bin, B_bin).sum()
        union = np.logical_or(A_bin, B_bin).sum()
        return intersection / (union + 1e-8)
    elif method == "dice":
        # Dice Coefficient
        thr = 0.5
        A_bin = (A > thr).astype(int)
        B_bin = (B > thr).astype(int)
        intersection = np.logical_and(A_bin, B_bin).sum()
        return 2 * intersection / (A_bin.sum() + B_bin.sum() + 1e-8)
    elif method == "ssim":
        # Structural Similarity Index (SSIM)
        score, _ = ssim(A, B, full=True, data_range=A.max() - A.min())
        return score
    elif method == "wasserstein":
        # Earth Mover's Distance (Wasserstein Distance)
        return wasserstein_distance(A.flatten(), B.flatten())
    else:
        raise ValueError(f"Unknown method: {method}")

def ssim(A, B):
    """
    Calculate the Structural Similarity Index (SSIM) between two matrices A and B.

    Parameters:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.

    Returns:
        float: SSIM score.
    """
    from skimage.metrics import structural_similarity as ssim
    score, _ = ssim(A, B, full=True, data_range=A.max() - A.min())
    return score

def wasserstein_distance(A, B):
    """
    Calculate the Earth Mover's Distance (Wasserstein Distance) between two matrices A and B.

    Parameters:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.

    Returns:
        float: Wasserstein distance score.
    """
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(A.flatten(), B.flatten())