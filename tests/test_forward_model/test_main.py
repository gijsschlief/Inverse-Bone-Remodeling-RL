from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

from forward_model.main import forward_model


def test_forward_model_valid_inputs():
    force_profile = np.zeros((3, 40))
    force_profile[0, 9] = 3
    initial_density = np.full((40, 40), 0.8)
    time_steps = 100
    dt = 1.0
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fenics"
    parameters = {
        'file_location': str(default_dir),
        'rho_min': 0.01,
        'rho_max': 1.74,
        'tolerance': 1E-14,
        'save': False,
        'plot': False
    }

    final_density = forward_model(force_profile, initial_density, time_steps, dt, parameters)
    assert isinstance(final_density, np.ndarray)
    assert final_density.shape == initial_density.shape

def test_forward_model_invalid_force_profile():
    force_profile = "invalid_force_profile"  # Invalid type
    initial_density = np.full((40, 40), 0.8)
    time_steps = 100
    dt = 1.0
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fenics"
    parameters = {
        'file_location': str(default_dir),
        'rho_min': 0.01,
        'rho_max': 1.74,
        'tolerance': 1E-14,
        'save': False,
        'plot': False
    }

    with pytest.raises(TypeError):
        forward_model(force_profile, initial_density, time_steps, dt, parameters)

def test_forward_model_invalid_initial_density():
    force_profile = np.zeros((3, 40))
    initial_density = "invalid_initial_density"  # Invalid type
    time_steps = 100
    dt = 1.0
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fenics"
    parameters = {
        'file_location': str(default_dir),
        'rho_min': 0.01,
        'rho_max': 1.74,
        'tolerance': 1E-14,
        'save': False,
        'plot': False
    }

    with pytest.raises(TypeError):
        forward_model(force_profile, initial_density, time_steps, dt, parameters)

def test_forward_model_invalid_time_steps():
    force_profile = np.zeros((3, 40))
    initial_density = np.full((40, 40), 0.8)
    time_steps = "invalid_time_steps"  # Invalid type
    dt = 1.0
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fenics"
    parameters = {
        'file_location': str(default_dir),
        'rho_min': 0.01,
        'rho_max': 1.74,
        'tolerance': 1E-14,
        'save': False,
        'plot': False
    }

    with pytest.raises(ValueError):
        forward_model(force_profile, initial_density, time_steps, dt, parameters)

def test_forward_model_invalid_dt():
    force_profile = np.zeros((3, 40))
    initial_density = np.full((40, 40), 0.8)
    time_steps = 100
    dt = "invalid_dt"  # Invalid type
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fenics"
    parameters = {
        'file_location': str(default_dir),
        'rho_min': 0.01,
        'rho_max': 1.74,
        'tolerance': 1E-14,
        'save': False,
        'plot': False
    }

    with pytest.raises(ValueError):
        forward_model(force_profile, initial_density, time_steps, dt, parameters)

def test_forward_model_default_parameters():
    force_profile = np.zeros((3, 40))
    initial_density = np.full((40, 40), 0.8)
    time_steps = 100
    dt = 1.0

    final_density = forward_model(force_profile, initial_density, time_steps, dt)
    assert isinstance(final_density, np.ndarray)
    assert final_density.shape == initial_density.shape

def test_main_valid_arguments():
    script_path = Path(__file__).resolve().parent.parent.parent / "forward_model" / "main.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--time_steps", "50", "--dt", "0.5", "--rho_min", "0.02", "--rho_max", "1.5"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Simulation completed." in result.stdout

def test_main_invalid_time_steps():
    script_path = Path(__file__).resolve().parent.parent.parent / "forward_model" / "main.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--time_steps", "invalid"],
        capture_output=True,
        text=True
    )
    assert result.returncode != 0
    assert "invalid int value" in result.stderr

def test_main_invalid_dt():
    script_path = Path(__file__).resolve().parent.parent.parent / "forward_model" / "main.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--dt", "invalid"],
        capture_output=True,
        text=True
    )
    assert result.returncode != 0
    assert "invalid float value" in result.stderr

def test_main_missing_file_location():
    script_path = Path(__file__).resolve().parent.parent.parent / "forward_model" / "main.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--file_location", "/non/existent/path"],
        capture_output=True,
        text=True
    )
    assert result.returncode != 0
    assert "No such file or directory" in result.stderr

def test_main_plot_flag():
    script_path = Path(__file__).resolve().parent.parent.parent / "forward_model" / "main.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--plot"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Simulation completed." in result.stdout

def test_forward_model_plot_enabled():
    force_profile = np.zeros((3, 40))
    initial_density = np.full((40, 40), 0.8)
    time_steps = 100
    dt = 1.0
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fenics"
    parameters = {
        'file_location': str(default_dir),
        'rho_min': 0.01,
        'rho_max': 1.74,
        'tolerance': 1E-14,
        'save': False,
        'plot': True  # Enable plotting
    }

    final_density = forward_model(force_profile, initial_density, time_steps, dt, parameters)
    assert isinstance(final_density, np.ndarray)
    assert final_density.shape == initial_density.shape

def test_forward_model_save_enabled():
    force_profile = np.zeros((3, 40))
    initial_density = np.full((40, 40), 0.8)
    time_steps = 100
    dt = 1.0
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fenics"
    parameters = {
        'file_location': str(default_dir),
        'rho_min': 0.01,
        'rho_max': 1.74,
        'tolerance': 1E-14,
        'save': True,  # Enable saving
        'plot': False
    }

    final_density = forward_model(force_profile, initial_density, time_steps, dt, parameters)
    assert isinstance(final_density, np.ndarray)
    assert final_density.shape == initial_density.shape
    
if __name__ == "__main__":
    pytest.main([__file__])