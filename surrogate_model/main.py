import os
import logging
import random

import numpy as np
import torch

from forward_model.data_reader import forward_data_reader
from rl_model.reward_calculation import calculate_similarity
from surrogate_model.neural_networks.advanced_neural_network import AdvancedNNSurrogateModel
from surrogate_model.loader import load_surrogate_model
from surrogate_model.splitting import splitting
from surrogate_model.visualize import compare_surrogate_model

def main():
    """
    Main function to load data, preprocess it, load the surrogate model, and evaluate its performance.
    """
    # Load the surrogate model and data
    model = load_surrogate_model("/home/gijs/Desktop/Thesis/data/models/trained_model.pth", AdvancedNNSurrogateModel)
    _, force_profiles, final_output_densities = forward_data_reader("/home/gijs/Desktop/Thesis/data/raw/training_batch_unknown_samples_25000_0605_0234.json")
    _, X_val, _, _, y_val, _ = splitting(force_profiles, final_output_densities)

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

    compare_surrogate_model(val_logits, y_val_tensor, sample_count=3, show_plot=True)
    return

if __name__ == "__main__":
    main()