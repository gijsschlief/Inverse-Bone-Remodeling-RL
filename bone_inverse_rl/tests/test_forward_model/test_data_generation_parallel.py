import os
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

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

def test_serialize_data():
    """Test the serialize_data function."""
    serial_number = 1
    force_profile = np.array([[0.5, -0.2], [0.1, 0.0]])
    result = np.array([[1.0, 1.0], [1.0, 1.0]])
    error = "Test error"

    serialized = serialize_data(serial_number, force_profile, result, error)

    assert serialized["serial_number"] == [serial_number]
    assert serialized["force_profile"] == force_profile.tolist()
    assert serialized["final_output_density"] == result.tolist()
    assert serialized["error"] == error

def test_serialize_data_no_error():
    """Test serialize_data when no error is provided."""
    serial_number = 1
    force_profile = np.array([[0.5, -0.2], [0.1, 0.0]])
    result = np.array([[1.0, 1.0], [1.0, 1.0]])

    serialized = serialize_data(serial_number, force_profile, result)

    assert serialized["serial_number"] == [serial_number]
    assert serialized["force_profile"] == force_profile.tolist()
    assert serialized["final_output_density"] == result.tolist()
    assert "error" not in serialized

def test_run_one_sample(mock_forward_model):
    """Test the _run_one_sample function."""
    args = (0, "/tmp", np.full((10, 10), 0.8), 100, 1.0, {"rho_min": 0.01, "rho_max": 1.74})
    result = _run_one_sample(args)

    assert result["serial_number"] == [1]
    assert isinstance(result["force_profile"], list)
    assert isinstance(result["final_output_density"], list)
    assert "error" not in result

def test_run_one_sample_with_error():
    """Test _run_one_sample when forward_model raises an exception."""
    with patch("Thesis_code.bone_inverse_rl.forward_model.main.forward_model", side_effect=Exception("Test error")):
        args = (0, "/tmp", np.full((10, 10), 0.8), 100, 1.0, {"rho_min": 0.01, "rho_max": 1.74})
        result = _run_one_sample(args)

        assert result["serial_number"] == [1]
        assert isinstance(result["force_profile"], list)
        assert isinstance(result["final_output_density"], list)
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