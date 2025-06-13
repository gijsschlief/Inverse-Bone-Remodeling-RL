import torch
import numpy as np
from surrogate_model.advanced_neural_network import AdvancedNNSurrogateModel
from utils.datareader import read_json_data
from utils.convert_to_array import convert_to_array
from utils.data_splitting import split_data
from rl_model.reward_calculation import calculate_similarity
import random

class SurrogateModelLoader:
    def __init__(self, model_path, model_class):
        """
        Initialize the loader with the path to the model and the model class.

        Args:
            model_path (str): Path to the .pth file containing the model weights.
            model_class (torch.nn.Module): The class of the model to be loaded.
        """
        self.model_path = model_path
        self.model_class = model_class
        self.model = None

    def load_model(self):
        """
        Load the surrogate model from the .pth file.
        """
        self.model = self.model_class()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()  # Set the model to evaluation mode

    def forward(self, data_points):
        """
        Run forward estimation on the given data points.

        Args:
            data_points (torch.Tensor): Input data points for the model.

        Returns:
            torch.Tensor: Model predictions.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
        
        with torch.no_grad():  # Disable gradient computation for inference
            predictions = self.model(data_points)
        return predictions
    
def main():
    # Data Loading
    data = read_json_data("/home/gijs/Desktop/Thesis/data/raw/training_data_20250605_023414.json")
    print(f"Data loaded")

    # Preprocessing
    serial_numbers, force_profiles, final_output_densities = convert_to_array(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(force_profiles, final_output_densities)

    # Load the surrogate model
    model_loader = SurrogateModelLoader("/home/gijs/Desktop/Thesis/data/models/trained_model.pth", AdvancedNNSurrogateModel)
    model_loader.load_model()

    # Evaluate the model on validation data
    model = model_loader.model  # Use the loaded model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    model.to(device)  # Move the model to the device
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        num_val_samples = X_val.shape[0]
        X_val_tensor = torch.tensor(X_val.reshape(num_val_samples, 3, 10).astype(np.float32)).to(device)
        y_val_tensor = torch.tensor(y_val.reshape(num_val_samples, 10, 10).astype(np.float32)).to(device)

        val_logits = model(X_val_tensor)
        loss_fn = torch.nn.MSELoss()  # Define the loss function
        val_loss = loss_fn(val_logits, y_val_tensor)

        print(f"Validation Loss: {val_loss.item()}")

    # Calculate similarity between predicted and actual validation data
    similarity_scores = []
    for i in range(num_val_samples):
        predicted_matrix = val_logits[i].cpu().numpy()
        average_similarity = np.mean(similarity_scores)
        actual_matrix = y_val_tensor[i].cpu().numpy()
        similarity = calculate_similarity(predicted_matrix, actual_matrix, method='wasserstein', baseline=0.1, threshold=0.5)
        similarity_scores.append(similarity)

    # Calculate average similarity as accuracy metric\
    #print(f"Similarity Scores: {similarity_scores}")
    average_similarity = np.mean(similarity_scores)
    print(f"Model Accuracy (Average Similarity): {average_similarity}")

    # print a matrix of the first validation sample
    print("First Validation Sample Predicted Matrix:")
    # Select 3 random indices from the validation set
    random_indices = random.sample(range(num_val_samples), 3)

    # Find the maximum and minimum values in the true results (actual matrices)
    max_true_value = torch.max(y_val_tensor).item()
    min_true_value = torch.min(y_val_tensor).item()

    for idx in random_indices:
        predicted_matrix = val_logits[idx].cpu().numpy()
        actual_matrix = y_val_tensor[idx].cpu().numpy()

        import matplotlib.pyplot as plt

        print(f"Sample Index: {idx}")
        print("Original Density Matrix:")
        print(actual_matrix)
        plt.figure(figsize=(6, 6))
        plt.imshow(actual_matrix, cmap='viridis', interpolation='nearest', vmin=min_true_value, vmax=max_true_value)
        plt.colorbar(label='Value')
        plt.title('Original Density Matrix')
        for i in range(actual_matrix.shape[0]):
            for j in range(actual_matrix.shape[1]):
                plt.text(j, i, f"{actual_matrix[i, j]:.2f}", ha='center', va='center', color='white', fontsize=8)
        plt.show()

        print("Predicted Density Matrix:")
        print(predicted_matrix)
        plt.figure(figsize=(6, 6))
        plt.imshow(predicted_matrix, cmap='viridis', interpolation='nearest', vmin=min_true_value, vmax=max_true_value)
        plt.colorbar(label='Value')
        plt.title('Predicted Density Matrix')
        for i in range(predicted_matrix.shape[0]):
            for j in range(predicted_matrix.shape[1]):
                plt.text(j, i, f"{predicted_matrix[i, j]:.2f}", ha='center', va='center', color='white', fontsize=8)
        plt.show()

        print("-" * 50)
    return

if __name__ == "__main__":
    main()