import torch
from bone_inverse_rl.surrogate_model.neural_network import NeuralNetwork
from bone_inverse_rl.utils.datareader import read_json_data
from bone_inverse_rl.utils.convert_to_array import convert_to_array
import numpy as np
from bone_inverse_rl.utils.data_splitting import split_data

# Data Loading
data = read_json_data("/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/data/raw/training_data_test_large.json")
print(f"Data loaded")

# Preprocessing
serial_numbers, force_profiles, final_output_densities = convert_to_array(data)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(force_profiles, final_output_densities)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
num_train_samples = X_train.shape[0]  # Calculate the number of training samples

X = torch.tensor(X_train.reshape(num_train_samples, 3, 10).astype(np.float32)).to(device)
y = torch.tensor(y_train.reshape(num_train_samples, 10, 10).astype(np.float32)).to(device)
print(f"Data converted and using {device} device and {num_train_samples} training samples")

model = NeuralNetwork().to(device)
print(f"Loaded the model: {model}")


# Pass the input through the model
logits = model(X)
pred_probab = torch.nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
# Define the loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()  # Set the model to training mode

    # Forward pass
    logits = model(X)
    loss = loss_fn(logits, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for every epoch
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Save the trained model
model.save_model("trained_model.pth")
