import numpy as np
import pytest
from bone_inverse_rl.forward_model.forward_model_input_test import forward_model_input_test

def test_forward_model_input_test_valid_inputs():
    force_profile = np.random.rand(10, 10)
    initial_density = np.full((10, 10), 0.8)
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': 0.01,
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error is None, "Expected no error for valid inputs."

def test_forward_model_input_test_invalid_force_profile_type():
    force_profile = "invalid_type"  # Not a numpy array
    initial_density = np.full((10, 10), 0.8)
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': 0.01,
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("TypeError", "force_profile must be a numpy array."), "Expected TypeError for invalid force_profile type."

def test_forward_model_input_test_invalid_initial_density_type():
    force_profile = np.random.rand(10, 10)
    initial_density = "invalid_type"  # Not a numpy array
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': 0.01,
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("TypeError", "initial_density must be a numpy array."), "Expected TypeError for invalid initial_density type."

def test_forward_model_input_test_invalid_time_steps_type():
    force_profile = np.random.rand(10, 10)
    initial_density = np.full((10, 10), 0.8)
    time_steps = "invalid_type"  # Not an integer
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': 0.01,
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("ValueError", "time_steps must be a positive integer."), "Expected ValueError for invalid time_steps type."

def test_forward_model_input_test_invalid_dt_type():
    force_profile = np.random.rand(10, 10)
    initial_density = np.full((10, 10), 0.8)
    time_steps = 100
    dt = "invalid_type"  # Not a number
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': 0.01,
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("ValueError", "dt must be a positive number."), "Expected ValueError for invalid dt type."

def test_forward_model_input_test_invalid_file_location_type():
    force_profile = np.random.rand(10, 10)
    initial_density = np.full((10, 10), 0.8)
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 12345,  # Not a string
        'rho_min': 0.01,
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("TypeError", "file_location must be a string."), "Expected TypeError for invalid file_location type."

def test_forward_model_input_test_invalid_rho_min_type():
    force_profile = np.random.rand(10, 10)
    initial_density = np.full((10, 10), 0.8)
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': "invalid_type",  # Not a number
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("TypeError", "rho_min must be a number."), "Expected TypeError for invalid rho_min type."

def test_forward_model_input_test_invalid_rho_max_type():
    force_profile = np.random.rand(10, 10)
    initial_density = np.full((10, 10), 0.8)
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': 0.01,
        'rho_max': "invalid_type",  # Not a number
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("TypeError", "rho_max must be a number."), "Expected TypeError for invalid rho_max type."

def test_forward_model_input_test_invalid_rho_min_value():
    force_profile = np.random.rand(10, 10)
    initial_density = np.full((10, 10), 0.8)
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': -0.01,  # Negative value
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("ValueError", "rho_min must be non-negative and rho_max must be greater than rho_min."), "Expected ValueError for invalid rho_min value."

def test_forward_model_input_test_invalid_rho_max_value():
    force_profile = np.random.rand(10, 10)
    initial_density = np.full((10, 10), 0.8)
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': 0.01,
        'rho_max': 0.005,  # Less than rho_min
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("ValueError", "rho_min must be non-negative and rho_max must be greater than rho_min."), "Expected ValueError for invalid rho_max value."

def test_forward_model_input_test_invalid_force_profile_shape():
    force_profile = np.random.rand(10, 10)
    initial_density = np.full((5, 5), 0.8)  # Different shape
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': 0.01,
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("ValueError", "force_profile and initial_density must have the same shape."), "Expected ValueError for different shapes."

def test_forward_model_input_test_invalid_initial_density_values():
    force_profile = np.random.rand(10, 10)
    initial_density = np.full((10, 10), -0.1)  # Negative values
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': 0.01,
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("ValueError", "initial_density values must be between 0 and 1."), "Expected ValueError for invalid initial_density values."

def test_forward_model_input_test_invalid_force_profile_values():
    force_profile = np.full((10, 10), -0.1)  # Negative values
    initial_density = np.full((10, 10), 0.8)
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': 0.01,
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("ValueError", "force_profile values must be non-negative."), "Expected ValueError for invalid force_profile values."

def test_forward_model_input_test_invalid_rho_min_range():
    force_profile = np.random.rand(10, 10)
    initial_density = np.full((10, 10), 0.8)
    time_steps = 100
    dt = 1.0
    parameters = {
        'file_location': 'bone_inverse_rl/data/raw',
        'rho_min': -0.01,  # Negative value
        'rho_max': 1.74,
    }
    
    error = forward_model_input_test(force_profile, initial_density, time_steps, dt, parameters)
    assert error == ("ValueError", "rho_min must be non-negative and rho_max must be greater than rho_min."), "Expected ValueError for invalid rho_min range."

if __name__ == "__main__":
    pytest.main([__file__])