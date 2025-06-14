import os
import json
import numpy as np
import pytest
import unittest

from fenics import set_log_level, LogLevel

from forward_model.data_generation import TrainingDataGenerator

def training_data_generator(temp_output_dir):
    """Fixture to create a TrainingDataGenerator instance."""
    return TrainingDataGenerator(output_dir=temp_output_dir, force_max=2, force_count_max=7, batch_seed=42)

def test_generate_parallel_creates_file(training_data_generator):
    """Test if generate_parallel creates a JSON file with the correct number of samples."""
    num_samples = 5
    training_data_generator.generate_parallel(num_samples)

    # Check if a file is created in the output directory
    files = [f for f in os.listdir(training_data_generator.output_dir) if f.startswith("training_batch_") and f.endswith(".json")]
    assert len(files) == 1, f"Expected 1 file, but found {len(files)} files in the output directory."

    # Check if the file contains valid JSON data
    filepath = os.path.join(training_data_generator.output_dir, files[0])
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    assert isinstance(data, list), "Data should be a list."
    assert len(data) == num_samples, f"Expected {num_samples} samples, but got {len(data)}."

def test_generate_sequential_creates_file(training_data_generator):
    """Test if generate_sequential creates a JSON file with the correct number of samples."""
    num_samples = 5
    training_data_generator.generate_sequential(num_samples)

    # Check if a file is created in the output directory
    files = [f for f in os.listdir(training_data_generator.output_dir) if f.startswith("training_batch_") and f.endswith(".json")]
    assert len(files) == 1, f"Expected 1 file, but found {len(files)} files in the output directory."

    # Check if the file contains valid JSON data
    filepath = os.path.join(training_data_generator.output_dir, files[0])
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    assert isinstance(data, list), "Data should be a list."
    assert len(data) == num_samples, f"Expected {num_samples} samples, but got {len(data)}."

def test_generate_edge_cases_creates_file(training_data_generator):
    """Test if generate_edge_cases creates a JSON file with the correct number of samples."""
    num_samples = 5
    training_data_generator.generate_edge_cases(num_samples)

    # Check if a file is created in the output directory
    files = [f for f in os.listdir(training_data_generator.output_dir) if f.startswith("edge_case_batch_") and f.endswith(".json")]
    assert len(files) == 1, f"Expected 1 file, but found {len(files)} files in the output directory."

    # Check if the file contains valid JSON data
    filepath = os.path.join(training_data_generator.output_dir, files[0])
    with open(filepath, 'r') as json_file:
        data = json.load(json_file)
    assert isinstance(data, list), "Data should be a list."
    assert len(data) == num_samples, f"Expected {num_samples} samples, but got {len(data)}."

def test_generate_random_force_profile(training_data_generator):
    """Test if _generate_random_force_profile generates valid force profiles."""
    sample_index = 0
    force_profile = training_data_generator._generate_random_force_profile(sample_index)

    assert force_profile.shape == (3, max(training_data_generator.initial_density.shape)), "Force profile shape is incorrect."
    assert np.any(force_profile != 0), "Force profile should contain non-zero values."

def test_run_sample_handles_exceptions(training_data_generator):
    """Test if _run_sample handles exceptions during forward_model execution."""
    force_profile = np.zeros((3, max(training_data_generator.initial_density.shape)))

    # Mock the forward_model function to raise an exception
    with unittest.mock.patch("forward_model.main.forward_model", side_effect=Exception("Mocked error")):
        result = training_data_generator._run_sample((0, force_profile))

    assert np.isnan(np.array(result["result"])).all(), "Result should be NaN."
    assert "error" in result, "Error field should be present in the result."
    assert result["error"] == "Mocked error", "Error message is incorrect."
    
if __name__ == "__main__":
    pytest.main([__file__])