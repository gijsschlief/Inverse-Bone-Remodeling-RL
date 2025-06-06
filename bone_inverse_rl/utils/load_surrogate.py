import torch

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
    
if __name__ == "__main__":
    # Example usage
    import torch
    import numpy as np
    # Removed unused import for NNSurrogateModel
    from bone_inverse_rl.surrogate_model.advanced_neural_network import AdvancedNNSurrogateModel
    from bone_inverse_rl.utils.datareader import read_json_data
    from bone_inverse_rl.utils.convert_to_array import convert_to_array
    from bone_inverse_rl.utils.data_splitting import split_data
    from bone_inverse_rl.rl_model.reward_calculation import calculate_similarity

    # Data Loading
    data = read_json_data("/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/data/raw/training_data_20250605_023414.json")
    print(f"Data loaded")

    # Preprocessing
    serial_numbers, force_profiles, final_output_densities = convert_to_array(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(force_profiles, final_output_densities)
    
    # Load the surrogate model
    model_loader = SurrogateModelLoader("Thesis_code/bone_inverse_rl/surrogate_model/trained_model.pth", AdvancedNNSurrogateModel)
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
        average_similarity = np.mean(similarity_scores)()
        actual_matrix = y_val_tensor[i].cpu().numpy()
        similarity = calculate_similarity(predicted_matrix, actual_matrix)
        similarity_scores.append(similarity)

    # Calculate average similarity as accuracy metric
    average_similarity = np.mean(similarity_scores)
    print(f"Model Accuracy (Average Similarity): {average_similarity}")
