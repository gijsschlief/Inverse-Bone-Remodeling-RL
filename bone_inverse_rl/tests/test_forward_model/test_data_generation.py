import os
import json
import numpy as np
import pytest
import unittest

from fenics import set_log_level, LogLevel

from Thesis_code.bone_inverse_rl.forward_model.data_generation import generate_training_data, serialize_data

@pytest.fixture
def temp_output_dir(tmpdir):
    """Fixture to create a temporary output directory."""
    return str(tmpdir)

def test_generate_training_data_creates_file(temp_output_dir):
    """Test if generate_training_data creates a JSON file in the specified directory."""
    set_log_level(LogLevel.ERROR)
    num_samples = 3
    generate_training_data(output_dir=temp_output_dir, num_samples=num_samples)

    # Check if a file is created in the output directory
    files = [f for f in os.listdir(temp_output_dir) if f.startswith("training_data_") and f.endswith(".json")]
    assert len(files) == 1, f"Expected 1 file, but found {len(files)} files in the output directory."
    
    # Check if the file contains valid JSON data
    filepath = os.path.join(temp_output_dir, files[0])
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    assert isinstance(data, list), "Data should be a list."
    assert len(data) == num_samples, f"Expected {num_samples} samples, but got {len(data)}."

def test_generate_training_data_handles_exceptions(temp_output_dir):
    """Test if generate_training_data handles exceptions during forward_model execution."""
    num_samples = 5

    # Mock the forward_model function to raise an exception
    with unittest.mock.patch("Thesis_code.bone_inverse_rl.forward_model.data_generation.forward_model", side_effect=Exception("Mocked error")):
        generate_training_data(output_dir=temp_output_dir, num_samples=num_samples)

    # Check if the file contains valid JSON data
    files = os.listdir(temp_output_dir)
    filepath = os.path.join(temp_output_dir, files[0])
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)

    # Verify that all samples have NaN outputs and error messages
    for sample in data:
        assert np.isnan(np.array(sample["final_output_density"])).all(), "Output density should be NaN."
        assert "error" in sample, "Error field should be present in the sample."

def test_serialize_data():
    """Test the serialize_data function."""
    serial_number = 1
    force_profile = np.array([[1.0, 0.0], [0.0, -1.0]])
    result = np.array([[0.8, 0.8], [0.8, 0.8]])
    error = "Test error"

    serialized = serialize_data(serial_number, force_profile, result, error)

    assert serialized["serial_number"] == serial_number, "Serial number mismatch."
    assert serialized["force_profile"] == force_profile.tolist(), "Force profile mismatch."
    assert serialized["final_output_density"] == result.tolist(), "Final output density mismatch."
    assert serialized["error"] == error, "Error mismatch."

def test_serialize_data_no_error():
    """Test the serialize_data function when no error is provided."""
    serial_number = 1
    force_profile = np.array([[1.0, 0.0], [0.0, -1.0]])
    result = np.array([[0.8, 0.8], [0.8, 0.8]])

    serialized = serialize_data(serial_number, force_profile, result)

    assert serialized["serial_number"] == serial_number, "Serial number mismatch."
    assert serialized["force_profile"] == force_profile.tolist(), "Force profile mismatch."
    assert serialized["final_output_density"] == result.tolist(), "Final output density mismatch."
    assert "error" not in serialized, "Error field should not be present."

if __name__ == "__main__":
    pytest.main([__file__])