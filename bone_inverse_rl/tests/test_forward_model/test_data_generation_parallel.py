import json
from unittest.mock import patch
import os

import numpy as np
import pytest
from multiprocessing import cpu_count
from fenics import set_log_level, LogLevel

from Thesis_code.bone_inverse_rl.forward_model.data_generation_parallel import (
    _run_one_sample,
    generate_training_data,
    serialize_data,
)

@pytest.fixture
def mock_forward_model():
    """Mock the forward_model function."""
    with patch("Thesis_code.bone_inverse_rl.forward_model.main.forward_model") as mock:
        mock.return_value = np.full((10, 10), 1.0)  # Mock output
        yield mock

def test_run_one_sample(mock_forward_model):
    """Test the _run_one_sample function."""
    args = (0, "/tmp", np.full((10, 10), 0.8), 100, 1.0, {"rho_min": 0.01, "rho_max": 1.74})
    result = _run_one_sample(args)

    assert result["serial_number"] == 1
    assert isinstance(result["force_profile"], list)
    assert isinstance(result["final_output_density"], list)
    assert "error" not in result

def test_run_one_sample_with_error():
    """Test _run_one_sample when forward_model raises an exception."""
    with patch("Thesis_code.bone_inverse_rl.forward_model.main.forward_model", side_effect=Exception("Test error")):
        args = (0, "/tmp", np.full((10, 10), 0.8), 100, 1.0, {"rho_min": 0.01, "rho_max": 1.74})
        result = _run_one_sample(args)

        assert result["serial_number"] == 1
        assert isinstance(result["force_profile"], list)
        assert isinstance(result["final_output_density"], list)
        if "error" in result:
            assert result["error"] == "Test error"

def test_generate_training_data(mock_forward_model, tmpdir):
    """Test the generate_training_data function."""
    output_dir = tmpdir.mkdir("output")
    generate_training_data(output_dir=str(output_dir), num_samples=5)

    # Check if the file is created
    files = list(output_dir.listdir())
    assert len(files) == 1
    assert files[0].ext == ".json"

    # Check the contents of the file
    with open(files[0], "r") as f:
        data = json.load(f)
        assert len(data) == 5
        for entry in data:
            assert "serial_number" in entry
            assert "force_profile" in entry
            assert "final_output_density" in entry

def test_generate_training_data_parallel(tmpdir):
    """Test the parallel variant of generate_training_data."""
    set_log_level(LogLevel.ERROR)
    num_samples = 6
    # Create a temporary output directory
    output_dir = tmpdir.mkdir("output")

    # Call the parallel version of generate_training_data
    generate_training_data(output_dir=str(output_dir), num_samples=num_samples)

    # Check if a file is created in the output directory
    files = [f for f in output_dir.listdir() if f.basename.startswith("training_data_") and f.basename.endswith(".json")]
    assert len(files) == 1, f"Expected 1 file, but found {len(files)} files in the output directory."

    # Check if the file contains valid JSON data
    filepath = str(files[0])
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    assert isinstance(data, list), "Data should be a list."
    assert len(data) == num_samples, f"Expected {num_samples} samples, but got {len(data)}."

    # Verify that all samples have the required fields
    for sample in data:
        assert "serial_number" in sample, "Missing serial_number field."
        assert "force_profile" in sample, "Missing force_profile field."
        assert "final_output_density" in sample, "Missing final_output_density field."
        
if __name__ == "__main__":
    pytest.main([__file__])
