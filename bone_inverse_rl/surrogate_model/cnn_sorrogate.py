import torch
from bone_inverse_rl.surrogate_model.neural_network import NeuralNetwork
from bone_inverse_rl.utils.datareader import read_json_data
from bone_inverse_rl.utils.convert_to_array import convert_to_array
import numpy as np

data = read_json_data("/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/data/raw/training_data_test.json")


serial_numbers, force_profiles, final_output_densities = convert_to_array(data)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

# Ensure the input tensor is converted to a PyTorch tensor and moved to the correct device
X = torch.tensor(force_profiles.reshape(100, 3, 10).astype(np.float32)).to(device)

# Ensure the target tensor matches the model's output shape
y = torch.tensor(final_output_densities.reshape(100, 10, 10).astype(np.float32)).to(device)

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
