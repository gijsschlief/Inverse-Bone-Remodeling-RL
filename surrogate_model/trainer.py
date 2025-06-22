"""Trainer script for the SurrogateModel."""

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from bone_remodeling.forward_model.data_reader import forward_data_reader
from bone_remodeling.rl_model.reward_calculation import calculate_similarity
from bone_remodeling.surrogate_model.neural_networks.large_nn import (
    LargeSurrogateModel,
)
from bone_remodeling.surrogate_model.neural_networks.medium_nn import (
    MediumSurrogateModel,
)
from bone_remodeling.surrogate_model.normalizor import (
    normalize_data,
    save_normalization_params,
)
from bone_remodeling.surrogate_model.splitter import splitting
from pytorch_msssim import ssim

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 50
MIN_DELTA = 1e-4
MODEL_PATH = "/home/gijs/Desktop/Thesis/data/models/trained_model.pth"
DATA_FILE_PATH = "/home/gijs/Desktop/Thesis/data/raw/"
NORMALIZE = True  # Set to False if you want to skip normalization


def load_data(
    path_pattern: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess data using forward_data_reader, which handles directories and checks.

    Args:
    ----
        path_pattern (str): Path to the JSON file or directory.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_train: Training features
            - X_val: Validation features
            - X_test: Test features
            - y_train: Training labels
            - y_val: Validation labels
            - y_test: Test labels

    Raises:
    ------
        ValueError: If data loading fails or if the input path is invalid.
        AssertionError: If the loaded data does not match expected dimensions.

    """
    result = forward_data_reader(path_pattern)
    if result is None:
        raise ValueError("Data loading failed. Please check the input path.")
    _, force_profiles, final_output_densities = result
    if force_profiles is None or final_output_densities is None:
        raise ValueError("Data loading failed. Please check the input path.")
    X = force_profiles
    y = final_output_densities
    return splitting(X, y, random_state=0)


def sanitize_data(
    x_data: np.ndarray, y_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Sanitize the input data by filtering out rows with NaN values in either x_data or y_data.

    Args:
    ----
        x_data (np.ndarray): Input features to be sanitized.
        y_data (np.ndarray): Target labels to be sanitized.

    Returns:
    -------
        Tuple[np.ndarray, np.ndarray]: Sanitized x_data and y_data with NaN rows removed.

    Raises:
    ------
        ValueError: If the input data is not a numpy array.

    """
    if not isinstance(x_data, np.ndarray) or not isinstance(y_data, np.ndarray):
        raise ValueError("Both x_data and y_data must be numpy arrays.")
    # Flatten all but the first dimension to check for NaNs per sample
    x_mask = ~np.isnan(x_data.reshape(x_data.shape[0], -1)).any(axis=1)
    y_mask = ~np.isnan(y_data.reshape(y_data.shape[0], -1)).any(axis=1)
    mask = x_mask & y_mask
    removed = x_data.shape[0] - np.count_nonzero(mask)
    if removed > 0:
        logging.error(f"sanitize_data: Removed {removed} datapoints due to NaN values.")
    return x_data[mask], y_data[mask]


def prepare_tensors(
    X_data: np.ndarray | torch.Tensor,
    y_data: np.ndarray | torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare input and output tensors for the model.

    Args:
    ----
        X_data (np.ndarray or torch.Tensor): Input features.
        y_data (np.ndarray or torch.Tensor): Target labels.
        device (torch.device): Device to which tensors will be moved.

    Returns:
    -------
        Tuple[torch.Tensor, torch.Tensor]: Input and output tensors reshaped for the model.

    Raises:
    ------
        ValueError: If the input data is not in the expected shape.

    """
    num_samples = X_data.shape[0]

    # Handle X
    if isinstance(X_data, torch.Tensor):
        X_tensor = X_data.clone().detach().to(torch.float32).reshape(num_samples, 3, 10)
    else:
        X_tensor = torch.tensor(X_data, dtype=torch.float32).reshape(num_samples, 3, 10)

    # Handle y
    if isinstance(y_data, torch.Tensor):
        y_tensor = y_data.clone().detach().to(torch.float32).reshape(num_samples, 10, 10)
    else:
        y_tensor = torch.tensor(y_data, dtype=torch.float32).reshape(num_samples, 10, 10)

    return X_tensor.to(device), y_tensor.to(device)



def ssim_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the Structural Similarity Index (SSIM) loss between predicted and target tensors.

    Args:
    ----
        predicted (torch.Tensor): Predicted output tensor.
        target (torch.Tensor): Target output tensor.

    Returns:
    -------
        torch.Tensor: SSIM loss value.

    """
    predicted = predicted.unsqueeze(1)
    target = target.unsqueeze(1)

    return 1 - ssim(predicted, target, win_size=3, data_range=target.max() - target.min())

def combined_loss(predicted: torch.Tensor, target: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Weighted combination of MSE and SSIM losses."""
    mse = torch.nn.functional.mse_loss(predicted, target)
    ssim_l = ssim_loss(predicted, target)
    return alpha * mse + (1 - alpha) * ssim_l

def train_model(
    model: MediumSurrogateModel,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
    min_delta: float = MIN_DELTA,
) -> None:
    """Train the SurrogateModel with early stopping and learning rate scheduling.

    Args:
    ----
        model (SurrogateModel): The model to be trained.
        X_train (torch.Tensor): Training input features.
        y_train (torch.Tensor): Training target labels.
        X_val (torch.Tensor): Validation input features.
        y_val (torch.Tensor): Validation target labels.
        device (torch.device): Device to which tensors will be moved.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        lr (float): Learning rate for the optimizer.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.

    """
    loss_fn = combined_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = model.get_scheduler(optimizer, epochs)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in model.create_dataloader(
            X_train, y_train, batch_size=batch_size
        ):
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            if torch.isnan(loss):
                logging.error("Loss is NaN, skipping this batch.")
                continue
            if torch.isinf(loss):
                logging.error("Loss is Inf, skipping this batch.")
                continue
            if not torch.isfinite(loss):
                logging.error("Loss is not finite, skipping this batch.")
                continue
            else:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_train_loss = total_loss / len(
            model.create_dataloader(X_train, y_train, batch_size=batch_size)
        )
        model.train_losses.append(avg_train_loss)

        best_model_state = None

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = combined_loss(val_logits, y_val)
        model.val_losses.append(val_loss)
        scheduler.step(val_loss)
        # Early stopping logic
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            logging.info(
                f"Epoch {epoch + 1}: Validation loss improved to {val_loss:.4f}. Saving model state."
            )
        else:
            epochs_no_improve += 1

        # Log every epoch for the first 10, then every 10 epochs
        if epoch < 10 or (epoch + 1) % 10 == 0:
            logging.info(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

        if epochs_no_improve >= patience:
            logging.info(
            f"Early stopping at epoch {epoch} (no improvement in {patience} epochs)."
            )
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info("Loaded best model state after training.")

def save_model_safely(model: MediumSurrogateModel, path: str) -> Path:
    """Save the model to a file, ensuring no overwriting of existing files."""
    if os.path.exists(path):
        base_path, ext = os.path.splitext(path)
        counter = 1
        while os.path.exists(f"{base_path}_{counter}{ext}"):
            counter += 1
        path = f"{base_path}_{counter}{ext}"
    model.save_model(path)
    return Path(path)


def evaluate_model(
    model: MediumSurrogateModel, X_val: torch.Tensor, y_val: torch.Tensor
) -> None:
    """Evaluate the model on the validation set and log the results."""
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = combined_loss(val_logits, y_val).item()

    similarities = [
        calculate_similarity(val_logits[i].cpu().numpy(), y_val[i].cpu().numpy())
        for i in range(X_val.shape[0])
    ]
    average_similarity = np.mean(similarities)
    logging.info(
        f"Validation Loss: {val_loss:.4f}, Average Similarity: {average_similarity:.4f}"
    )


def main() -> None:
    """Train and evaluate the surrogate model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loading data from {DATA_FILE_PATH}")
    X_train_np, X_val_np, X_test_np, y_train_np, y_val_np, y_test_np = load_data(
        DATA_FILE_PATH
    )

    logging.info("Sanitizing data...")
    X_train_np, y_train_np = sanitize_data(X_train_np, y_train_np)
    X_val_np, y_val_np = sanitize_data(X_val_np, y_val_np)
    X_test_np, y_test_np = sanitize_data(X_test_np, y_test_np)

    if NORMALIZE:
        logging.info("Normalizing data...")
        X_train, X_val, X_test, X_mean, X_std  = normalize_data(X_train_np, X_val_np, X_test_np)
        y_train, y_val, y_test, y_mean, y_std = normalize_data(y_train_np, y_val_np, y_test_np)
        logging.info(f"Normalization parameters: X_mean={X_mean}, X_std={X_std}, y_mean={y_mean}, y_std={y_std}")

    logging.info("Preparing tensors...")
    X_train, y_train = prepare_tensors(X_train, y_train, device)
    X_val, y_val = prepare_tensors(X_val, y_val, device)
    X_test, y_test = prepare_tensors(X_test, y_test, device)

    model = LargeSurrogateModel().to(device)
    logging.info(f"Model architecture:\n{model}")

    logging.info(
        f"Training on {len(X_train)} samples, validating on {len(X_val)} samples."
    )
    train_model(model, X_train, y_train, X_val, y_val, device)

    model_path = save_model_safely(model, MODEL_PATH)
    if NORMALIZE:
        save_normalization_params(
            model_path.with_suffix(".npz"),
            X_mean,
            X_std,
            y_mean,
            y_std,
        )
    logging.info(f"Model saved to {model_path}")

    model.plot_loss()

    logging.info("Evaluating model on validation set.")

    evaluate_model(model, X_val, y_val)
    logging.info("Training and evaluation complete.")


if __name__ == "__main__":
    main()
