"""Trainer script for the InverseSurrogateModel."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bone_remodeling.src.forward_data.visualizer import plot_density_matrix
from bone_remodeling.src.inverse_model.inverse_neural_network_simple import (
    InverseModel,
)
from bone_remodeling.src.inverse_model.train_parameters import (
    InverseTrainParameters,
)
from bone_remodeling.src.rl_model.reward_calculation import calculate_similarity
from bone_remodeling.src.surrogate_model.ensemble import (
    load_ensemble_models,
    predict_with_ensemble,
)
from bone_remodeling.src.surrogate_model.loader import SurrogateModelLoader
from bone_remodeling.src.surrogate_model.neural_networks.reversed_nn import (
    ReversedSurrogateModel,
)
from bone_remodeling.src.surrogate_model.normalizor import (
    normalize_data,
    save_normalization_params,
)
from bone_remodeling.src.surrogate_model.sanitizer import sanitize_data
from bone_remodeling.src.surrogate_model.splitter import load_and_split_data

logger = logging.getLogger(__name__)


def convert_tensors(
    x_data: np.ndarray | torch.Tensor,
    y_data: np.ndarray | torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare input and output tensors for the model.

    Args:
    ----
        x_data (np.ndarray or torch.Tensor): Input features (N, int, int, float).
        y_data (np.ndarray or torch.Tensor): Target labels (N, 3, 10).
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
        x_tensor = x_data.clone().detach().to(torch.float32).reshape(num_samples, 10, 10)
    else:
        x_tensor = torch.tensor(x_data, dtype=torch.float32).reshape(num_samples, 10, 10)

    # Handle y
    if isinstance(y_data, torch.Tensor):
        y_tensor = (
            y_data.clone().detach().to(torch.float32).reshape(num_samples, 3)
        )
    else:
        y_tensor = torch.tensor(y_data, dtype=torch.float32).reshape(
            num_samples,
            3,
        )

    return x_tensor.to(device), y_tensor.to(device)


def train_model(
    model: InverseModel,
    x_data: tuple[torch.Tensor, torch.Tensor],
    y_data: tuple[torch.Tensor, torch.Tensor],
    train_parameters: InverseTrainParameters,
) -> None:
    """Train the InverseModel with early stopping and learning rate scheduling.

    Args:
    ----
        model (InverseModel): The model to be trained.
        x_data (tuple[torch.Tensor, torch.Tensor]): Input features and validation features.
        y_data (tuple[torch.Tensor, torch.Tensor]): Target labels and validation labels.
        train_parameters (InverseTrainParameters): Training parameters including device, epochs, batch size, learning rate, patience, min delta, log interval, and log all for first epochs.

    """
    x_train, x_validation = x_data
    y_train, y_validation = y_data

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_parameters.learning_rate)
    scheduler = model.get_scheduler(optimizer, train_parameters.epochs)

    best_val_loss = float("inf")
    epochs_no_improve = 0

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

        avg_train_loss = total_loss / len(
            model.create_dataloader(
                x_train,
                y_train,
                batch_size=train_parameters.batch_size,
            ),
        )
        model.train_losses.append(avg_train_loss)

        best_model_state = None

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


def save_model_safely(model: InverseModel, path: Path) -> Path:
    """Save the model to a file, ensuring no overwriting of existing files."""
    if path.exists():
        base_path = path.stem
        ext = path.suffix
        counter = 1
        new_path = path.with_name(f"{base_path}_{counter}{ext}")
        while new_path.exists():
            counter += 1
            new_path = path.with_name(f"{base_path}_{counter}{ext}")
        path = new_path
    model.save_model(path)
    return path


def combined_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the combined loss (currently MSE)."""
    return torch.nn.functional.mse_loss(predictions, targets)

def evaluate_model(
    model: InverseModel,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
) -> None:
    """Evaluate the inverse model by comparing original vs. reconstructed density profiles.

    The process:
    1. Predict force parameters from density (inverse model).
    2. Reconstruct force profiles from parameters.
    3. Run the ensemble surrogate model to predict densities from forces.
    4. Compare reconstructed densities with original densities using SSIM.
    """
    model.eval()
    with torch.no_grad():
        validation_predictions = model(x_validation)
        validation_loss = combined_loss(validation_predictions, y_validation).item()

    # Convert predicted parameters → 3x10 force profiles
    validation_predictions = validation_predictions.cpu().numpy()
    reconstructed_forces = np.array([
        params_to_force_profile(
            int(np.clip(pred[0], 0, 9)),
            int(np.clip(np.round(pred[1]), 0, 2)),
            float(np.clip(pred[2], 0.0, 1.0)),
        )
        for pred in validation_predictions
    ])

    # Load the trained forward ensemble models (force → density)
    model_paths = [
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_1.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_2.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_3.pth"),
        Path("/home/gijs/Desktop/Thesis/data/models/trained_model_4.pth"),
    ]
    ensemble_models, x_means, x_stds, y_means, y_stds = load_ensemble_models(model_paths, model_class=ReversedSurrogateModel,
        model_loader=SurrogateModelLoader)

    # Predict densities from the estimated force profiles
    predicted_densities, _ = predict_with_ensemble(ensemble_models, reconstructed_forces, [x_means, x_stds], [y_means, y_stds])
    if isinstance(predicted_densities, torch.Tensor):
        predicted_densities = predicted_densities.cpu().numpy()

    # Compare reconstructed vs. true densities
    true_densities = x_validation.cpu().numpy()
    true_forces = y_validation.cpu().numpy()
    true_force_profiles = np.array([
            params_to_force_profile(
                int(np.clip(true_force_data[0], 0, 9)),
                int(np.clip(np.round(true_force_data[1]), 0, 2)),
                float(np.clip(true_force_data[2], 0.0, 1.0)),
            )
            for true_force_data in true_forces
        ])


    ssim_scores = []
    for i in range(len(true_densities)):
        ssim_score = calculate_similarity(
                reference_matrix=true_densities[i],
                comparison_matrix=predicted_densities[i],
                method="ssim",
                baseline=0.1,
                threshold=0.5)
        ssim_scores.append(ssim_score)

    average_similarity = float(np.mean(ssim_scores))
    logger.info(
        f"Validation Loss: {validation_loss:.4f} | Average SSIM (density reconstruction): {average_similarity:.4f}",
    )

    # === Show sample visualizations ===
    for i in range(10):
        fig, axes = plt.subplots(1, 2)
        idx = np.random.choice(len(true_densities))
        fig.suptitle(f"Inverse Model Evaluation Samples (Original vs. Prediction) {idx}")
        plot_density_matrix(
            true_densities[idx],
            force_profile=true_force_profiles[idx],
            axis=axes[0],
            title="Original data",
        )
        plot_density_matrix(
            predicted_densities[idx],
            force_profile=reconstructed_forces[idx],
            axis=axes[1],
            title="Prediction",
        )
        plt.tight_layout()
        plt.show()

def force_profile_to_params(force_profile: np.ndarray) -> tuple[int, int, float]:
    """Convert a 3xN force profile into peak location, peak side, and peak height."""
    if force_profile.shape[0] != 3:
        raise ValueError("Force profile must have shape (3, N)")

    peak_side = int(np.argmax(np.max(force_profile, axis=1)))
    peak_height = float(np.max(force_profile[peak_side]))
    peak_location = int(np.argmax(force_profile[peak_side]))

    return peak_location, peak_side, peak_height


def params_to_force_profile(
    peak_location: int,
    peak_side: int,
    peak_height: float,
    length: int = 10,
) -> np.ndarray:
    """Convert peak location, peak side, and peak height back into a triangular 3xN force profile."""
    peak_side = min(peak_side, 2)
    if not (0 <= peak_side < 3):
        raise ValueError("Peak side must be 0, 1, or 2")
    if not (0 <= peak_location < length):
        raise ValueError(f"Peak location must be between 0 and {length - 1}")

    force_profile = np.zeros((3, length), dtype=np.float32)

    for j in range(length):
        if j < peak_location and peak_location > 0:
            force_profile[peak_side, j] = peak_height * (j / peak_location)
        elif j > peak_location and peak_location < length - 1:
            force_profile[peak_side, j] = peak_height * ((length - 1 - j) / (length - 1 - peak_location))
        else:
            force_profile[peak_side, j] = peak_height
    return force_profile

def main(
    data_file_path: Path,
    model_path: Path,
    *,
    normalize: bool = True,
    random_state: int = 0,
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
    ) = load_and_split_data(data_file_path, random_state=random_state)

    logger.info("Sanitizing data...")
    x_train_np, y_train_np = sanitize_data(x_train_np, y_train_np)
    x_val_np, y_val_np = sanitize_data(x_val_np, y_val_np)
    x_test_np, y_test_np = sanitize_data(x_test_np, y_test_np)

    # Turn the force profiles into 3 datapoints: peak location, peak side, peak height
    x_train_np = np.array(
        [force_profile_to_params(fp.reshape(3, -1)) for fp in x_train_np],
        dtype=np.float32,
    )
    x_val_np = np.array(
        [force_profile_to_params(fp.reshape(3, -1)) for fp in x_val_np],
        dtype=np.float32,
    )
    x_test_np = np.array(
        [force_profile_to_params(fp.reshape(3, -1)) for fp in x_test_np],
        dtype=np.float32,
    )

    if normalize:
        logger.info("Normalizing data...")
        y_train_np, y_mean, y_std = normalize_data(y_train_np)
        y_val_np, _, _ = normalize_data(y_val_np, y_mean, y_std)
        y_test_np, _, _ = normalize_data(y_test_np, y_mean, y_std)

        logger.info(
            f"Normalization parameters: y_mean={y_mean}, y_std={y_std}",
        )
    else:
        logger.info("Skipping normalization.")

    # TURN THE NAMES AROUND
    y_train_tensor = torch.tensor(x_train_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(x_val_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(x_test_np, dtype=torch.float32)
    x_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    x_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)
    x_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

    logger.info("Preparing tensors...")
    x_train, y_train = convert_tensors(x_train_tensor, y_train_tensor, device)
    if x_val_tensor is not None and y_val_tensor is not None:
        x_val, y_val = convert_tensors(x_val_tensor, y_val_tensor, device)
    if x_test_tensor is not None and y_test_tensor is not None:
        x_test, y_test = convert_tensors(x_test_tensor, y_test_tensor, device)

    model = InverseModel().to(device)
    logger.info(f"Model architecture:\n{model}")

    logger.info(
        f"Training on {len(x_train)} samples, validating on {len(x_val) if x_val is not None else 0} samples.",
    )
    train_model(
        model,
        (x_train, x_val),
        (y_train, y_val),
        train_parameters=InverseTrainParameters(
            device=device,
        ),
    )

    model_path = save_model_safely(model, model_path)
    if normalize:
        save_normalization_params(
            model_path.with_suffix(".npz"),
            y_mean,
            y_std,
            0,
            0,
        )
    logger.info(f"Model saved to {model_path}")

    # model.plot_loss()

    logger.info("Evaluating model on validation set.")

    evaluate_model(model, x_val, y_val)
    logger.info("Training and evaluation complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # Show INFO and above
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_path = Path("/home/gijs/Desktop/Thesis/data/inverse_model/trained_model.pth")
    data_file_path = Path("/home/gijs/Desktop/Thesis/data/raw/triangular/")

    main(data_file_path, model_path, normalize=False, random_state=1)

    logger.info("All training runs completed.")
    logger.info("Final model saved at: %s", model_path)
