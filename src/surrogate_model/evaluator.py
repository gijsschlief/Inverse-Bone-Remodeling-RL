"""Evaluator for Surrogate Model."""

import logging

import numpy as np
import torch

from bone_remodeling.src.rl_model.reward_calculation import calculate_similarity

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def validate_surrogate_model(
    model: torch.nn.Module,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[np.ndarray, np.ndarray]:
    """Run the surrogate model on validation data and return predicted and true matrices.

    This function takes a trained surrogate model and validation data, runs the model to get predictions,
    and returns the predicted matrices and the true matrices. It ensures that the input data is in the correct
    format and reshapes it appropriately for the model. It also checks that the model is a PyTorch module and
    moves it to the specified device if necessary.

    Args:
    ----
        model: The trained surrogate model.
        x_val: Validation input data.
        y_val: Validation target data.
        device: The device to run the model on (default is CUDA if available, otherwise CPU).

    Raises:
    ------
        ValueError: If the input data is not in the correct format or if the model is not a torch.nn.Module.

    Returns:
    -------
        tuple[np.ndarray, np.ndarray]: A tuple containing the predicted matrices and the true matrices.

    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(x_val, np.ndarray) or not isinstance(y_val, np.ndarray):
        raise ValueError("Both X_val and y_val must be numpy.ndarray objects.")

    if x_val.shape[0] != y_val.shape[0]:
        logging.warning(
            "X_val and y_val have different number of samples. Using the minimum of both."
        )
        num_samples = np.min([x_val.shape[0], y_val.shape[0]])
        x_val = x_val[:num_samples]
        y_val = y_val[:num_samples]
    else:
        num_samples = x_val.shape[0]
    if not isinstance(model, torch.nn.Module):
        raise ValueError("The model must be an instance of torch.nn.Module.")
    if next(model.parameters()).device != device:
        model.to(device)
    model.eval()

    with torch.no_grad():
        # Validate that X_val can be reshaped to (num_samples, 3, 10)
        if x_val.size != num_samples * 3 * 10:
            raise ValueError(
                f"x_val with shape {x_val.shape} cannot be reshaped to ({num_samples}, 3, 10)."
            )
        # Validate that y_val can be reshaped to (num_samples, 10, 10)
        if y_val.size != num_samples * 10 * 10:
            raise ValueError(
                f"y_val with shape {y_val.shape} cannot be reshaped to ({num_samples}, 10, 10)."
                "Ensure y_val has the correct number of elements."
            )
        y_val_tensor = torch.from_numpy(
            y_val.reshape(num_samples, 10, 10).astype(np.float32)
        ).to(device)
        x_val_tensor = torch.from_numpy(
            x_val.reshape(num_samples, 3, 10).astype(np.float32)
        ).to(device)

    # Run the model to get predictions
    val_logits: torch.Tensor = model(x_val_tensor)
    predicted_matrices = val_logits.detach().cpu().numpy()
    true_matrices = y_val_tensor.detach().cpu().numpy()
    return predicted_matrices, true_matrices


def average_similarity_score(
    predicted_matrices: np.ndarray,
    true_matrices: np.ndarray,
    num_samples: int | None = None,
    baseline: float = 0.1,
    threshold: float = 0.5,
    method: str = "ssim",
) -> float:
    """Calculate the similarity score between predicted and true matrices with error logging."""
    if not isinstance(predicted_matrices, np.ndarray) or not isinstance(
        true_matrices, np.ndarray
    ):
        logging.error(
            "Both predicted_matrices and true_matrices must be numpy.ndarray objects."
        )
        raise ValueError(
            "Both predicted_matrices and true_matrices must be numpy.ndarray objects."
        )

    if predicted_matrices.shape[1:] != true_matrices.shape[1:]:
        logging.error(
            f"Shape mismatch: predicted_matrices has shape {predicted_matrices.shape}, "
            f"true_matrices has shape {true_matrices.shape}."
        )
        raise ValueError(
            "predicted_matrices and true_matrices must have the same shape except for the first dimension."
        )

    if num_samples is None:
        num_samples = min(predicted_matrices.shape[0], true_matrices.shape[0])

    if (
        num_samples > predicted_matrices.shape[0]
        or num_samples > true_matrices.shape[0]
    ):
        logging.error(
            f"num_samples ({num_samples}) is greater than the number of available samples: "
            f"predicted_matrices ({predicted_matrices.shape[0]}), true_matrices ({true_matrices.shape[0]})."
        )
        raise ValueError(
            "num_samples exceeds the available number of samples in the input arrays."
        )

    similarity_scores: list = []

    for i in range(num_samples):
        try:
            similarity = calculate_similarity(
                predicted_matrices[i],
                true_matrices[i],
                method=method,
                baseline=baseline,
                threshold=threshold,
            )
            similarity_scores.append(similarity)
        except Exception as e:
            logging.error(
                f"Error calculating similarity for sample {i}: {e}. "
                f"predicted_matrix shape: {predicted_matrices[i].shape}, "
                f"true_matrix shape: {true_matrices[i].shape}"
            )
            raise

    if not similarity_scores:
        logging.error("No similarity scores were calculated. Check input data.")
        raise ValueError("No similarity scores calculated.")

    # Filter out NaN values before calculating the mean
    filtered_scores = [score for score in similarity_scores if not np.isnan(score)]
    if not filtered_scores:
        logging.error("All similarity scores are NaN. Check input data.")
        raise ValueError("All similarity scores are NaN.")
    mean_score = np.mean(filtered_scores)
    if np.isnan(mean_score):
        logging.error(
            "Mean similarity score is NaN. Check if similarity_scores contains valid values."
        )
        raise ValueError("Mean similarity score is NaN.")
    return float(mean_score)
