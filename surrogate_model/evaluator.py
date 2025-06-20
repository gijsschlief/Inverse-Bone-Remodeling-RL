"""Evaluator for Surrogate Model."""

import logging

import numpy as np
import torch

from rl_model.reward_calculation import calculate_similarity

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate_surrogate_model(
    model: torch.nn.Module, X_val: np.ndarray, y_val: np.ndarray
) -> tuple:
    """Evaluate the surrogate model on validation data and calculate similarity scores.

    This function takes a trained surrogate model and validation data, evaluates the model,
    and computes the similarity between the predicted and actual validation outputs.
    It returns the model's predictions, the actual validation outputs, and the average similarity score.
    The validation data should be in the form of numpy arrays, where `X_val` is the input data
    and `y_val` is the target data. The model should be a PyTorch neural network module.

    Args:
    ----
        model: The trained surrogate model.
        X_val: Validation input data.
        y_val: Validation target data.

    Raises:
    ------
        ValueError: If the input data is not in the correct format or if the model is not a torch.nn.Module.

    """
    if not isinstance(X_val, np.ndarray) or not isinstance(y_val, np.ndarray):
        raise ValueError("Both X_val and y_val must be numpy.ndarray objects.")

    if X_val.shape[0] != y_val.shape[0]:
        logging.warning(
            "X_val and y_val have different number of samples. Using the minimum of both."
        )
        num_samples = np.min([X_val.shape[0], y_val.shape[0]])
        X_val = X_val[:num_samples]
        y_val = y_val[:num_samples]
    else:
        num_samples = X_val.shape[0]
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Model must be of type torch.nn.Module")

    # Ensure the model is in evaluation mode and on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_val_tensor = torch.tensor(
            X_val.reshape(num_samples, 3, 10).astype(np.float32)
        ).to(device)
        y_val_tensor = torch.tensor(
            y_val.reshape(num_samples, 10, 10).astype(np.float32)
        ).to(device)

        val_logits = model(X_val_tensor)
        loss_fn = torch.nn.MSELoss()  # Define the loss function
        val_loss = loss_fn(val_logits, y_val_tensor)

        logging.info(f"Validation Loss: {val_loss.item()}")

    # Calculate similarity between predicted and actual validation data
    similarity_scores = []
    for i in range(num_samples):
        predicted_matrix = val_logits[i].cpu().numpy()
        actual_matrix = y_val_tensor[i].cpu().numpy()
        similarity = calculate_similarity(
            predicted_matrix, actual_matrix, method="ssim", baseline=0.1, threshold=0.5
        )
        similarity_scores.append(similarity)

    # Calculate average similarity as accuracy metric
    # print(f"Similarity Scores: {similarity_scores}")
    average_similarity = np.mean(similarity_scores)
    logging.info(f"Model Accuracy (Average Similarity): {average_similarity}")
    return val_logits, y_val_tensor, average_similarity
