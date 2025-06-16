import logging

import numpy as np
import torch

def evaluate_model(model: torch.nn.Module, X_val: np.ndarray, y_val):
    """
    Evaluate the surrogate model on validation data and calculate similarity scores.
    Args:
        model: The trained surrogate model.
        X_val: Validation input data.
        y_val: Validation target data.
    """
    # Ensure the model is in evaluation mode and on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        num_val_samples = X_val.shape[0]
        X_val_tensor = torch.tensor(X_val.reshape(num_val_samples, 3, 10).astype(np.float32)).to(device)
        y_val_tensor = torch.tensor(y_val.reshape(num_val_samples, 10, 10).astype(np.float32)).to(device)

        val_logits = model(X_val_tensor)
        loss_fn = torch.nn.MSELoss()  # Define the loss function
        val_loss = loss_fn(val_logits, y_val_tensor)

        logging.info(f"Validation Loss: {val_loss.item()}")

    # Calculate similarity between predicted and actual validation data
    similarity_scores = []
    for i in range(num_val_samples):
        predicted_matrix = val_logits[i].cpu().numpy()
        average_similarity = np.mean(similarity_scores)
        actual_matrix = y_val_tensor[i].cpu().numpy()
        similarity = calculate_similarity(predicted_matrix, actual_matrix, method='ssim', baseline=0.1, threshold=0.5)
        similarity_scores.append(similarity)

    # Calculate average similarity as accuracy metric\
    #print(f"Similarity Scores: {similarity_scores}")
    average_similarity = np.mean(similarity_scores)
    logging.info(f"Model Accuracy (Average Similarity): {average_similarity}")

if __name__ == "__main__":
    main()