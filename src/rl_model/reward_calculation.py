"""Calculate similarity or distance scores between two matrices using various methods."""

import warnings
from typing import Callable

import numpy as np
from scipy.stats import wasserstein_distance as wasserstein_calculation
from skimage.metrics import structural_similarity as _ssim

BASELINE_DEFAULT: float = 0.1
THRESHOLD_DEFAULT: float = 0.5
SUPPORTED_METHODS: list[str] = [
    "mse",
    "mae",
    "cosine",
    "iou",
    "dice",
    "ssim",
    "wasserstein",
]

_METHOD_FUNCS: dict[str, Callable[..., float]]


def calculate_similarity(
    ref: np.ndarray,
    cmp: np.ndarray,
    method: str = "mse",
    baseline: float = BASELINE_DEFAULT,
    threshold: float = THRESHOLD_DEFAULT,
) -> float:
    """Compute similarity or distance score between two matrices using the specified method.

    Args:
    ----
        ref (np.ndarray): Reference matrix.
        cmp (np.ndarray): Comparison matrix.
        method (str): One of SUPPORTED_METHODS.
        baseline (float): Baseline value for error normalization.
        threshold (float): Threshold value for binary overlap methods.

    Returns:
    -------
        float: Similarity or distance score normalized to [-1, 1] (except SSIM).

    """
    _validate_inputs(ref, cmp, method, baseline, threshold)
    func = _METHOD_FUNCS[method]
    return func(ref, cmp, baseline, threshold)


def _validate_inputs(
    ref: np.ndarray,
    cmp: np.ndarray,
    method: str,
    baseline: float,
    threshold: float,
) -> None:
    """Validate inputs to calculate_similarity."""
    if not isinstance(ref, np.ndarray) or not isinstance(cmp, np.ndarray):
        raise TypeError("Both inputs must be numpy arrays.")
    if ref.shape != cmp.shape or ref.size == 0:
        raise ValueError("Input matrices must have the same non-empty shape.")
    if np.all(ref == 0) and np.all(cmp == 0):
        raise ValueError("Both matrices cannot be all zeros.")
    if baseline <= 0:
        raise ValueError("Baseline must be positive.")
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method '{method}'.")
    if method not in ["mse", "mae", "wasserstein"] and baseline != BASELINE_DEFAULT:
        warnings.warn(
            f"Baseline unused for '{method}'. Resetting to default.",
            UserWarning,
            stacklevel=2,
        )
    if method not in ["iou", "dice"] and threshold != THRESHOLD_DEFAULT:
        warnings.warn(
            f"Threshold unused for '{method}'. Resetting to default.",
            UserWarning,
            stacklevel=2,
        )


def _norm_err(metric: float, baseline: float) -> float:
    """Normalize an error metric to [-1, 1]."""
    return 2 * (1 - metric / (metric + baseline)) - 1


def _norm_overlap(metric: float) -> float:
    """Normalize an overlap metric [0,1] to [-1, 1]."""
    return 2 * metric - 1


def _mse(ref: np.ndarray, cmp: np.ndarray, baseline: float, *_: float) -> float:
    metric: float = np.mean((ref - cmp) ** 2)
    return _norm_err(metric, baseline)


def _mae(ref: np.ndarray, cmp: np.ndarray, baseline: float, *_: float) -> float:
    metric: float = np.mean(np.abs(ref - cmp))
    return _norm_err(metric, baseline)


def _cosine(ref: np.ndarray, cmp: np.ndarray, *_: float) -> float:
    v1: np.ndarray = ref.ravel()
    v2: np.ndarray = cmp.ravel()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)


def _iou(ref: np.ndarray, cmp: np.ndarray, _: float, threshold: float) -> float:
    a: np.ndarray = ref > threshold
    b: np.ndarray = cmp > threshold
    intersection: int = np.logical_and(a, b).sum()
    union: int = np.logical_or(a, b).sum()
    return _norm_overlap(intersection / (union + 1e-8))


def _dice(ref: np.ndarray, cmp: np.ndarray, _: float, threshold: float) -> float:
    a: np.ndarray = ref > threshold
    b: np.ndarray = cmp > threshold
    intersection: int = np.logical_and(a, b).sum()
    total: int = a.sum() + b.sum()
    return _norm_overlap(2 * intersection / (total + 1e-8))


def _ssim_score(ref: np.ndarray, cmp: np.ndarray, *_: float) -> float:
    score: float
    score, _ = _ssim(ref, cmp, full=True, data_range=ref.max() - ref.min())
    return score


def _wasserstein(
    ref: np.ndarray,
    cmp: np.ndarray,
    baseline: float,
    *_: float,
) -> float:
    distances: list[float] = [
        wasserstein_calculation(ref[i], cmp[i]) for i in range(ref.shape[0])
    ]
    metric: float = float(np.mean(distances))
    return _norm_err(metric, baseline)


_METHOD_FUNCS = {
    "mse": _mse,
    "mae": _mae,
    "cosine": _cosine,
    "iou": _iou,
    "dice": _dice,
    "ssim": _ssim_score,
    "wasserstein": _wasserstein,
}
