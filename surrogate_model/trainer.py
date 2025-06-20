"""Trainer script for the AdvancedNNSurrogateModel."""

import logging
import os
from typing import Tuple

import numpy as np
import torch
from bone_remodeling.forward_model.data_reader import forward_data_reader
from bone_remodeling.rl_model.reward_calculation import calculate_similarity
from bone_remodeling.surrogate_model.neural_networks.advanced_neural_network import (
    AdvancedNNSurrogateModel,
)
from bone_remodeling.surrogate_model.splitter import splitting

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 20
MIN_DELTA = 1e-4
MODEL_PATH = "/home/gijs/Desktop/Thesis/data/models/trained_model.pth"
DATA_FILE_PATH = "/home/gijs/Desktop/Thesis/data/raw/"


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
    return splitting(X, y)


def prepare_tensors(
    X_data: np.ndarray, y_data: np.ndarray, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare input and output tensors for the model.

    Args:
    ----
        X_data (np.ndarray): Input features.
        y_data (np.ndarray): Target labels.
        device (torch.device): Device to which tensors will be moved.

    Returns:
    -------
        Tuple[torch.Tensor, torch.Tensor]: Input and output tensors reshaped for the model.

    Raises:
    ------
        ValueError: If the input data is not in the expected shape.

    """
    num_samples = X_data.shape[0]
    X_tensor = torch.tensor(X_data.reshape(num_samples, 3, 10).astype(np.float32)).to(
        device
    )
    y_tensor = torch.tensor(y_data.reshape(num_samples, 10, 10).astype(np.float32)).to(
        device
    )
    return X_tensor, y_tensor


def train_model(
    model: AdvancedNNSurrogateModel,
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
    """Train the AdvancedNNSurrogateModel with early stopping and learning rate scheduling.

    Args:
    ----
        model (AdvancedNNSurrogateModel): The model to be trained.
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
    loss_fn = torch.nn.MSELoss()
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
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(
            model.create_dataloader(X_train, y_train, batch_size=batch_size)
        )
        model.train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val)
        model.val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Early stopping logic
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            logging.info(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

        if epochs_no_improve >= patience:
            logging.info(
                f"Early stopping at epoch {epoch} (no improvement in {patience} epochs)."
            )
            break


def save_model_safely(model: AdvancedNNSurrogateModel, path: str) -> None:
    """Save the model to a file, ensuring no overwriting of existing files."""
    if os.path.exists(path):
        base_path, ext = os.path.splitext(path)
        counter = 1
        while os.path.exists(f"{base_path}_{counter}{ext}"):
            counter += 1
        path = f"{base_path}_{counter}{ext}"
    model.save_model(path)
    logging.info(f"Model saved to {path}")


def evaluate_model(
    model: AdvancedNNSurrogateModel, X_val: torch.Tensor, y_val: torch.Tensor
) -> None:
    """Evaluate the model on the validation set and log the results."""
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = torch.nn.MSELoss()(val_logits, y_val).item()

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

    X_train, y_train = prepare_tensors(X_train_np, y_train_np, device)
    X_val, y_val = prepare_tensors(X_val_np, y_val_np, device)

    model = AdvancedNNSurrogateModel().to(device)
    logging.info(f"Model architecture:\n{model}")

    logging.info(
        f"Training on {len(X_train)} samples, validating on {len(X_val)} samples."
    )
    train_model(model, X_train, y_train, X_val, y_val, device)

    model.plot_loss()

    save_model_safely(model, MODEL_PATH)

    logging.info("Evaluating model on validation set.")

    evaluate_model(model, X_val, y_val)
    logging.info("Training and evaluation complete.")


if __name__ == "__main__":
    main()
