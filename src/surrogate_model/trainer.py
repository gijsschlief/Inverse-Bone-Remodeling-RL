"""Trainer script for the SurrogateModel."""

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from bone_remodeling.src.rl_model.reward_calculation import calculate_similarity
from bone_remodeling.src.surrogate_model.loss_function import combined_loss
from bone_remodeling.src.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)
from bone_remodeling.src.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.src.surrogate_model.normalizor import (
    normalize_data,
    save_normalization_params,
)
from bone_remodeling.src.surrogate_model.sanitizer import sanitize_data
from bone_remodeling.src.surrogate_model.splitter import load_and_split_data

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def convert_tensors(
    x_data: np.ndarray | torch.Tensor,
    y_data: np.ndarray | torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare input and output tensors for the model.

    Args:
    ----
        x_data (np.ndarray or torch.Tensor): Input features.
        y_data (np.ndarray or torch.Tensor): Target labels.
        device (torch.device): Device to which tensors will be moved.

    Returns:
    -------
        Tuple[torch.Tensor, torch.Tensor]: Input and output tensors reshaped for the model.

    Raises:
    ------
        ValueError: If the input data is not in the expected shape.

    """
    num_samples = x_data.shape[0]

    # Handle X
    if isinstance(x_data, torch.Tensor):
        x_tensor = x_data.clone().detach().to(torch.float32).reshape(num_samples, 3, 10)
    else:
        x_tensor = torch.tensor(x_data, dtype=torch.float32).reshape(num_samples, 3, 10)

    # Handle y
    if isinstance(y_data, torch.Tensor):
        y_tensor = (
            y_data.clone().detach().to(torch.float32).reshape(num_samples, 10, 10)
        )
    else:
        y_tensor = torch.tensor(y_data, dtype=torch.float32).reshape(
            num_samples, 10, 10
        )

    return x_tensor.to(device), y_tensor.to(device)


def train_model(
    model: SurrogateModel,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 10,
    min_delta: float = 1e-4,
) -> None:
    """Train the SurrogateModel with early stopping and learning rate scheduling.

    Args:
    ----
        model (SurrogateModel): The model to be trained.
        x_train (torch.Tensor): Training input features.
        y_train (torch.Tensor): Training target labels.
        x_val (torch.Tensor): Validation input features.
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
        for batch_x, batch_y in model.create_dataloader(
            x_train, y_train, batch_size=batch_size
        ):
            optimizer.zero_grad()
            logits = model(batch_x)
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
            model.create_dataloader(x_train, y_train, batch_size=batch_size)
        )
        model.train_losses.append(avg_train_loss)

        best_model_state = None

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = combined_loss(val_logits, y_val)
        model.val_losses.append(val_loss)
        scheduler.step(val_loss)
        # Early stopping logic
        if val_loss + min_delta < best_val_loss:
            best_val_loss = float(val_loss)
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


def save_model_safely(model: SurrogateModel, path: Path) -> Path:
    """Save the model to a file, ensuring no overwriting of existing files."""
    if os.path.exists(path):
        base_path, ext = os.path.splitext(path)
        counter = 1
        while os.path.exists(f"{base_path}_{counter}{ext}"):
            counter += 1
        path = Path(f"{base_path}_{counter}{ext}")
    model.save_model(path)
    return Path(path)


def evaluate_model(
    model: SurrogateModel, x_val: torch.Tensor, y_val: torch.Tensor
) -> None:
    """Evaluate the model on the validation set and log the results."""
    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
        val_loss = combined_loss(val_logits, y_val).item()

    similarities = [
        calculate_similarity(val_logits[i].cpu().numpy(), y_val[i].cpu().numpy())
        for i in range(x_val.shape[0])
    ]
    average_similarity = np.mean(similarities)
    logging.info(
        f"Validation Loss: {val_loss:.4f}, Average Similarity: {average_similarity:.4f}"
    )


def main(data_file_path: Path, model_path: Path, normalize: bool = True, random_state: int = 0) -> None:
    """Train and evaluate the surrogate model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loading data from {data_file_path}")
    x_train_np, x_val_np, x_test_np, y_train_np, y_val_np, y_test_np = (
        load_and_split_data(data_file_path, random_state=random_state)
    )

    logging.info("Sanitizing data...")
    x_train_np, y_train_np = sanitize_data(x_train_np, y_train_np)
    x_val_np, y_val_np = sanitize_data(x_val_np, y_val_np)
    x_test_np, y_test_np = sanitize_data(x_test_np, y_test_np)

    if normalize:
        logging.info("Normalizing data...")
        x_train_np, x_mean, x_std = normalize_data(x_train_np)
        x_val_np, _, _ = normalize_data(x_val_np, x_mean, x_std)
        x_test_np, _, _ = normalize_data(x_test_np, x_mean, x_std)
        y_train_np, y_mean, y_std = normalize_data(y_train_np)
        y_val_np, _, _ = normalize_data(y_val_np, y_mean, y_std)
        y_test_np, _, _ = normalize_data(y_test_np, y_mean, y_std)

        logging.info(
            f"Normalization parameters: x_mean={x_mean}, x_std={x_std}, y_mean={y_mean}, y_std={y_std}"
        )
    else:
        logging.info("Skipping normalization.")

    x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val_np, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

    logging.info("Preparing tensors...")
    x_train, y_train = convert_tensors(x_train_tensor, y_train_tensor, device)
    if x_val_tensor is not None and y_val_tensor is not None:
        x_val, y_val = convert_tensors(x_val_tensor, y_val_tensor, device)
    if x_test_tensor is not None and y_test_tensor is not None:
        x_test, y_test = convert_tensors(x_test_tensor, y_test_tensor, device)

    model = ReversedSurrogateModel().to(device)
    logging.info(f"Model architecture:\n{model}")

    logging.info(
        f"Training on {len(x_train)} samples, validating on {len(x_val) if x_val is not None else 0} samples."
    )
    train_model(model, x_train, y_train, x_val, y_val, device)

    model_path = save_model_safely(model, model_path)
    if normalize:
        save_normalization_params(
            model_path.with_suffix(".npz"),
            x_mean,
            x_std,
            y_mean,
            y_std,
        )
    logging.info(f"Model saved to {model_path}")

    model.plot_loss()

    logging.info("Evaluating model on validation set.")

    evaluate_model(model, x_val, y_val)
    logging.info("Training and evaluation complete.")


if __name__ == "__main__":
    model_path = Path("/home/gijs/Desktop/Thesis/data/models/trained_model.pth")
    data_file_path = Path("/home/gijs/Desktop/Thesis/data/raw/")

    for i in range(5):
        main(data_file_path, model_path, normalize=True, random_state=i+1)

    logging.info("All training runs completed.")
    logging.info("Final model saved at: %s", model_path)
    # 4 and up are trained on all data
