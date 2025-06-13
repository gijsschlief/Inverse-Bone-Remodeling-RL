import os
import json
import numpy as np
import pytest
import unittest

from fenics import set_log_level, LogLevel

from forward_model.data_generation import generate_training_data

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
    with unittest.mock.patch("forward_model.data_generation.forward_model", side_effect=Exception("Mocked error")):
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

if __name__ == "__main__":
    pytest.main([__file__])