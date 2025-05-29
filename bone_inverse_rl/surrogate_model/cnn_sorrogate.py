from torch import accelerator, nn, rand
from bone_inverse_rl.surrogate_model.neural_network import NeuralNetwork
from bone_inverse_rl.utils.datareader import read_json_data
from bone_inverse_rl.utils.convert_to_array import convert_to_array
import numpy as np

data = read_json_data("/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/data/raw/training_data_test.json")


serial_numbers, force_profiles, final_output_densities = convert_to_array(data)

device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

X = data

X = rand(1, 30, device=device)  # Adjust the input size to match the network's expected input
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")