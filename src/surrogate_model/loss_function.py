"""Create loss function for the surrogate model."""

from pytorch_msssim import ssim  # type: ignore
from torch import Tensor, nn


def ssim_loss(estimated_output: Tensor, reference_output: Tensor) -> Tensor:
    """Calculate the Structural Similarity Index (SSIM) loss between predicted and target tensors.

    Args:
    ----
        estimated_output (torch.Tensor): Predicted output tensor.
        reference_output (torch.Tensor): Target output tensor.

    Returns:
    -------
        torch.Tensor: SSIM loss value.

    """
    estimated_output = estimated_output.unsqueeze(1)
    reference_output = reference_output.unsqueeze(1)

    return 1 - ssim(
        estimated_output,
        reference_output,
        win_size=3,
        data_range=reference_output.max() - reference_output.min(),
    )


def combined_loss(
    predicted: Tensor,
    target: Tensor,
    loss_weight: float = 0.5,
) -> Tensor:
    """Weighted combination of Mean Squared Error (MSE) and SSIM loss.

    Args:
    ----
        predicted (torch.Tensor): Predicted output tensor.
        target (torch.Tensor): Target output tensor.
        loss_weight (float): Weight for the MSE loss in the combined loss function (default is 0.5).

    Returns:
    -------
        torch.Tensor: Combined loss value.

    """
    mean_squared_error = nn.functional.mse_loss(predicted, target)
    ssim_loss_value = ssim_loss(predicted, target)
    return loss_weight * mean_squared_error + (1 - loss_weight) * ssim_loss_value
