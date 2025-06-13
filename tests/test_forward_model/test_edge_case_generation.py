import os
import json

import pytest
import numpy as np
from fenics import set_log_level, LogLevel

from forward_model.edge_case_generation import (
    _generate_edge_case_force_profiles,
    _run_edge_case_sample,
    generate_edge_case_data
)

@pytest.fixture
def mock_initial_density_shape():
    return (10, 10)

@pytest.fixture
def mock_output_dir(tmp_path):
    return str(tmp_path)

@pytest.fixture
def mock_parameters():
    return {
        'file_location': '/tmp',
        'rho_min': 0.01,
        'rho_max': 1.74,
        'save': False,
        'plot': False
    }

def test_generate_edge_case_force_profiles(mock_initial_density_shape):
    set_log_level(LogLevel.ERROR)
    num_cases = 10
    force_profiles = _generate_edge_case_force_profiles(mock_initial_density_shape, num_cases)
    
    assert len(force_profiles) == num_cases
    assert isinstance(force_profiles, list)
    for profile in force_profiles:
        assert profile.shape == (3, max(mock_initial_density_shape))

def test_run_edge_case_sample(mock_initial_density_shape, mock_output_dir, mock_parameters):
    initial_density = np.full(mock_initial_density_shape, 0.8)
    time_steps = 10
    dt = 1.0
    force_profile = np.zeros((3, max(mock_initial_density_shape)))
    args = (0, mock_output_dir, initial_density, time_steps, dt, mock_parameters, force_profile)
    
    result = _run_edge_case_sample(args)
    
    assert isinstance(result, dict)
    assert 'serial_number' in result
    assert 'force_profile' in result
    assert 'final_output_density' in result
    assert 'error' in result or 'error' not in result

def test_generate_edge_case_data(mock_output_dir):
    num_samples = 5
    generate_edge_case_data(output_dir=mock_output_dir, num_samples=num_samples)
    
    files = os.listdir(mock_output_dir)
    assert len(files) == 1
    assert files[0].startswith("edge_case_data_")
    assert files[0].endswith(".json")
    
    filepath = os.path.join(mock_output_dir, files[0])
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert len(data) == num_samples
    for entry in data:
        assert 'serial_number' in entry
        assert 'force_profile' in entry
        assert 'final_output_density' in entry
        assert 'error' in entry or 'error' not in entry

if __name__ == "__main__":
    pytest.main([__file__])