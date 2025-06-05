import torch
from bone_inverse_rl.surrogate_model.neural_network import NNSurrogateModel
from bone_inverse_rl.surrogate_model.advanced_neural_network import AdvancedNNSurrogateModel
from bone_inverse_rl.utils.datareader import read_json_data
from bone_inverse_rl.utils.convert_to_array import convert_to_array
import numpy as np
from bone_inverse_rl.utils.data_splitting import split_data
from bone_inverse_rl.rl_model.reward_calculation import calculate_similarity

# Data Loading
data = read_json_data("/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/data/raw/training_data_20250605_023414.json")
print(f"Data loaded")

# Preprocessing
serial_numbers, force_profiles, final_output_densities = convert_to_array(data)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(force_profiles, final_output_densities)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

num_train_samples = X_train.shape[0]  # Calculate the number of training samples
print(num_train_samples)
X = torch.tensor(X_train.reshape(num_train_samples, 3, 10).astype(np.float32)).to(device)
y = torch.tensor(y_train.reshape(num_train_samples, 10, 10).astype(np.float32)).to(device)
print(f"Data converted and using {device} device and {num_train_samples} training samples")

# Load the model
#model = NNSurrogateModel().to(device)
model = AdvancedNNSurrogateModel().to(device)  # Use the advanced model if needed
print(f"Loaded the model: {model}")


# Pass the input through the model
logits = model(X)
# Output the predicted values directly for regression
print(f"Predicted output: {logits}")
# Define the loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# Training loop
epochs = 300
scheduler = model.get_scheduler(optimizer, epochs=epochs)  # Get the learning rate scheduler


for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    for batch_X, batch_y in model.create_dataloader(X, y, batch_size=32):
        # Forward pass
        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(model.create_dataloader(X, y, batch_size=32))
    model.train_losses.append(avg_train_loss)

    # Validation step
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        num_val_samples = X_val.shape[0]
        X_val_tensor = torch.tensor(X_val.reshape(num_val_samples, 3, 10).astype(np.float32)).to(device)
        y_val_tensor = torch.tensor(y_val.reshape(num_val_samples, 10, 10).astype(np.float32)).to(device)

        val_logits = model(X_val_tensor)
        val_loss = loss_fn(val_logits, y_val_tensor)
    model.val_losses.append(val_loss)
    scheduler.step(val_loss)  # Step the scheduler with validation loss

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the trained model
model.save_model("/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/surrogate_model/trained_model.pth")
model.plot_loss()  # Plot the training and validation loss  

# Evaluate the model on validation data
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    num_val_samples = X_val.shape[0]
    X_val_tensor = torch.tensor(X_val.reshape(num_val_samples, 3, 10).astype(np.float32)).to(device)
    y_val_tensor = torch.tensor(y_val.reshape(num_val_samples, 10, 10).astype(np.float32)).to(device)

    val_logits = model(X_val_tensor)
    val_loss = loss_fn(val_logits, y_val_tensor)

    print(f"Validation Loss: {val_loss.item()}")
    
    # Calculate similarity between predicted and actual validation data
    similarity_scores = []
    for i in range(num_val_samples):
        predicted_matrix = val_logits[i].cpu().numpy()
        actual_matrix = y_val_tensor[i].cpu().numpy()
        similarity = calculate_similarity(predicted_matrix, actual_matrix)
        similarity_scores.append(similarity)

    # Calculate average similarity as accuracy metric
    average_similarity = np.mean(similarity_scores)
    print(f"Model Accuracy (Average Similarity): {average_similarity}")
