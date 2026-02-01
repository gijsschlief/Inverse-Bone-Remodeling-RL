"""Trainer script for the SurrogateModel. Note that SSIMS are calculated on normalized data here and thus lower than the true values."""

import logging
import time
from pathlib import Path

import numpy as np
import torch

from bone_remodelling.rl_model.reward_calculation import calculate_similarity
from bone_remodelling.surrogate_model.loss_function import combined_loss
from bone_remodelling.surrogate_model.neural_networks.neural_network import (
    SurrogateModel,
)
from bone_remodelling.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodelling.surrogate_model.normalizor import (
    normalize_data,
    save_normalization_params,
)
from bone_remodelling.surrogate_model.sanitizer import sanitize_data
from bone_remodelling.surrogate_model.splitter import load_and_split_data
from bone_remodelling.surrogate_model.train_parameters import (
    SurrogateTrainParameters,
)

logger = logging.getLogger(__name__)


def convert_tensors(
    x_data: np.ndarray | torch.Tensor,
    y_data: np.ndarray | torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare input and output tensors for the model.

    Args:
    ----
        x_data (np.ndarray or torch.Tensor): Input features.
        y_data (np.ndarray or torch.Tensor): Target labels.
        device (torch.device): Device to which tensors will be moved.

    Returns:
    -------
        tuple[torch.Tensor, torch.Tensor]: Input and output tensors reshaped for the model.

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
            num_samples,
            10,
            10,
        )

    return x_tensor.to(device), y_tensor.to(device)


def train_model(
    model: SurrogateModel,
    x_data: tuple[torch.Tensor, torch.Tensor],
    y_data: tuple[torch.Tensor, torch.Tensor],
    train_parameters: SurrogateTrainParameters,
) -> None:
    """Train the SurrogateModel with early stopping and learning rate scheduling.

    Args:
    ----
        model (SurrogateModel): The model to be trained.
        x_data (tuple[torch.Tensor, torch.Tensor]): Input features and validation features.
        y_data (tuple[torch.Tensor, torch.Tensor]): Target labels and validation labels.
        train_parameters (SurrogateTrainParameters): Training parameters including device, epochs, batch size, learning rate, patience, min delta, log interval, and log all for first epochs.

    """
    x_train, x_validation = x_data
    y_train, y_validation = y_data

    loss_fn = combined_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_parameters.learning_rate)
    scheduler = model.get_scheduler(optimizer, train_parameters)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_time = time.time()
    best_model_state = None
    batch_count = len(model.create_dataloader(x_train, y_train, batch_size=train_parameters.batch_size))

    for epoch in range(train_parameters.epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in model.create_dataloader(
            x_train,
            y_train,
            batch_size=train_parameters.batch_size,
        ):
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            if torch.isnan(loss):
                logger.error("Loss is NaN, skipping this batch.")
                continue
            if torch.isinf(loss):
                logger.error("Loss is Inf, skipping this batch.")
                continue
            if not torch.isfinite(loss):
                logger.error("Loss is not finite, skipping this batch.")
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / batch_count
        model.train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_validation)
            val_loss = combined_loss(val_logits, y_validation)
        model.val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Early stopping logic
        if val_loss + train_parameters.min_delta < best_val_loss:
            best_val_loss = float(val_loss)
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            logger.info(
                f"Epoch {epoch + 1}: Validation loss improved to {val_loss:.4f}. Saving model state.",
            )
        else:
            epochs_no_improve += 1

        # Log every epoch for the first 10, then every 10 epochs
        if (
            epoch < train_parameters.log_all_for_first_epochs
            or (epoch + 1) % train_parameters.log_interval == 0
        ):
            logger.info(
                f"Epoch {epoch + 1}/{train_parameters.epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}",
            )

        if epochs_no_improve >= train_parameters.patience:
            logger.info(
                f"Early stopping at epoch {epoch} (no improvement in {train_parameters.patience} epochs).",
            )
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model state after training.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds.")

def save_model_safely(model: SurrogateModel, path: Path) -> Path:
    """Save the model to a file, ensuring no overwriting of existing files."""
    try:
        if path.exists():
            base_path = path.with_suffix("")
            ext = path.suffix
            counter = 1
            while Path(f"{base_path}_{counter}{ext}").exists():
                counter += 1
            path = Path(f"{base_path}_{counter}{ext}")
    except Exception as e:
        logger.warning(f"Safely saving model failed ({e}), overwriting existing file.")
        pass
    model.save_model(path)
    return Path(path)

def evaluate_model(
    model: SurrogateModel,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
) -> None:
    """Evaluate the model on the validation set and log the results."""
    model.eval()
    with torch.no_grad():
        validation_predictions = model(x_validation)
        validation_loss = combined_loss(validation_predictions, y_validation).item()

    similarities = [
        calculate_similarity(
            validation_predictions[i].cpu().numpy(),
            y_validation[i].cpu().numpy(),
        )
        for i in range(x_validation.shape[0])
    ]
    average_similarity = np.mean(similarities)
    logger.info(
        f"Validation Loss: {validation_loss:.4f}, Average Similarity: {average_similarity:.4f}",
    )

def rescramble_for_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = np.random.randint(0, 1_000_000),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rescramble training and validation data for ensemble training.

    Args:
    ----
        x_train (np.ndarray): Training input features.
        y_train (np.ndarray): Training target labels.
        x_val (np.ndarray): Validation input features.
        y_val (np.ndarray): Validation target labels.
        random_state (int): Random seed for reproducibility.

    Returns:
    -------
        tuple[np.ndarray, np.ndarray]: Rescrambled training input features and target labels.

    """
    np.random.seed(random_state)
    x_train_and_val = np.concatenate((x_train, x_val), axis=0)
    y_train_and_val = np.concatenate((y_train, y_val), axis=0)
    perm = np.random.permutation(x_train_and_val.shape[0])
    x_train_and_val = x_train_and_val[perm]
    y_train_and_val = y_train_and_val[perm]
    split_index = x_train.shape[0]
    x_train_rescrambled = x_train_and_val[:split_index]
    y_train_rescrambled = y_train_and_val[:split_index]
    x_val_rescrambled = x_train_and_val[split_index:]
    y_val_rescrambled = y_train_and_val[split_index:]
    return x_train_rescrambled, y_train_rescrambled, x_val_rescrambled, y_val_rescrambled

def main(
    data_file_path: Path,
    model_path: Path,
    *,
    normalize: bool = True,
    random_state: int = 1,
) -> None:
    """Train and evaluate the surrogate model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading data from {data_file_path}")
    (
        x_train_np,
        x_val_np,
        x_test_np,
        y_train_np,
        y_val_np,
        y_test_np,
    ) = load_and_split_data(data_file_path, random_state=1) # KEEP FIXED FOR ENSEMBLE TRAINING

    x_train_np, y_train_np, x_val_np, y_val_np = rescramble_for_ensemble(
        x_train_np,
        y_train_np,
        x_val_np,
        y_val_np,
        random_state=random_state,
    )

    logger.info("Sanitizing data...")
    x_train_np, y_train_np = sanitize_data(x_train_np, y_train_np)
    x_val_np, y_val_np = sanitize_data(x_val_np, y_val_np)
    x_test_np, y_test_np = sanitize_data(x_test_np, y_test_np)

    if normalize:
        logger.info("Normalizing data...")
        x_train_np, x_mean, x_std = normalize_data(x_train_np)
        x_val_np, _, _ = normalize_data(x_val_np, x_mean, x_std)
        x_test_np, _, _ = normalize_data(x_test_np, x_mean, x_std)
        y_train_np, y_mean, y_std = normalize_data(y_train_np)
        y_val_np, _, _ = normalize_data(y_val_np, y_mean, y_std)
        y_test_np, _, _ = normalize_data(y_test_np, y_mean, y_std)

        logger.info(
            f"Normalization parameters: x_mean={x_mean}, x_std={x_std}, y_mean={y_mean}, y_std={y_std}",
        )
    else:
        logger.info("Skipping normalization.")

    x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val_np, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

    logger.info("Preparing tensors...")
    x_train, y_train = convert_tensors(x_train_tensor, y_train_tensor, device)
    if x_val_tensor is not None and y_val_tensor is not None:
        x_val, y_val = convert_tensors(x_val_tensor, y_val_tensor, device)
    if x_test_tensor is not None and y_test_tensor is not None:
        x_test, y_test = convert_tensors(x_test_tensor, y_test_tensor, device)

    model = ReversedSurrogateModel().to(device)
    logger.info(f"Model architecture:\n{model}")

    logger.info(
        f"Training on {len(x_train)} samples, validating on {len(x_val) if x_val is not None else 0} samples.",
    )
    train_model(
        model,
        [x_train, x_val],
        [y_train, y_val],
        train_parameters=SurrogateTrainParameters(
            device=device,
        ),
    )

    model_path = save_model_safely(model, model_path)
    if normalize:
        save_normalization_params(
            model_path.with_suffix(".npz"),
            x_mean,
            x_std,
            y_mean,
            y_std,
        )
    logger.info(f"Model saved to {model_path}")

    #model.plot_loss()

    logger.info("Evaluating model normalised on validation set.")

    evaluate_model(model, x_val, y_val)

    logger.info("Evaluating model normalised on test set.")
    evaluate_model(model, x_test, y_test)

    logger.info("Training and evaluation complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_path = Path(__file__).parent.parent.parent / Path("data", "models", "surrogate.pth")
    data_file_path = Path(__file__).parent.parent.parent / Path("data", "raw")

    for i in range(2, 6): # USE FOR ENSEMBLE TRAINING
        main(data_file_path, model_path, normalize=True, random_state=i)

    logger.info("All training runs completed.")
    logger.info("Final model saved at: %s", model_path)
