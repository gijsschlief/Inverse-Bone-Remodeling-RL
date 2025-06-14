import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from forward_model.run_density_simulation import main

@pytest.fixture
def parameters_file_path():
    return Path(__file__).resolve().parent / "parameters.json"

@pytest.fixture
def mock_forward_model():
    with patch("forward_model.run_density_simulation.forward_model") as mock_model:
        mock_model.return_value = np.full((40, 40), 1.0)  # Mock final density
        yield mock_model

@pytest.fixture
def mock_argparse():
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        yield mock_args

def test_reset_parameters(parameters_file_path, mock_argparse):
    # Mock arguments to simulate --reset flag
    mock_argparse.return_value = MagicMock(reset=True)

    # Create a dummy parameters file
    parameters_file_path.write_text(json.dumps({"dummy_key": "dummy_value"}))

    assert parameters_file_path.exists()

    # Run the main function
    main()

    # Check if the parameters file is deleted
    assert not parameters_file_path.exists()

def test_save_parameters(parameters_file_path, mock_argparse, mock_forward_model):
    # Mock arguments with some parameters
    mock_args = MagicMock(
        reset=False,
        time_steps=50,
        dt=0.5,
        rho_min=0.1,
        rho_max=1.0,
        tolerance=0.01,
        B=0.5,
        k=0.2,
        nu=0.3,
        M=1000,
        gamma=2.0,
        file_name="output",
        file_extension=".txt",
        save=True,
        plot=False,
        convergence_eps=0.001,
        file_location="data/",
    )
    mock_argparse.return_value = mock_args

    # Run the main function
    main()

    # Check if the parameters file is created and contains the correct data
    assert parameters_file_path.exists()
    with open(parameters_file_path, "r") as f:
        parameters = json.load(f)
        assert parameters["time_steps"] == 50
        assert parameters["dt"] == 0.5
        assert parameters["rho_min"] == 0.1
        assert parameters["rho_max"] == 1.0

def test_forward_model_execution(mock_argparse, mock_forward_model):
    # Mock arguments
    mock_args = MagicMock(
        reset=False,
        time_steps=100,
        dt=1.0,
        save=False,
        plot=False,
    )
    mock_argparse.return_value = mock_args

    # Run the main function
    main()

    # Check if the forward_model function was called with correct arguments
    mock_forward_model.assert_called_once()
    args, kwargs = mock_forward_model.call_args
    assert args[0].shape == (3, 40)  # Force profile
    assert args[1].shape == (40, 40)  # Initial density
    assert args[2] == 100  # Time steps
    assert args[3] == 1.0  # Time step size