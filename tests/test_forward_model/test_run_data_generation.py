import pytest
from unittest.mock import patch, MagicMock
import argparse
import numpy as np
from pathlib import Path
from forward_model.run_data_generation import main

@pytest.fixture
def mock_training_data_generator():
    with patch("run_data_generation.TrainingDataGenerator") as mock_generator:
        instance = MagicMock()
        mock_generator.return_value = instance
        yield instance

@pytest.fixture
def mock_argparse():
    with patch("argparse.ArgumentParser.parse_args") as mock_args:
        args = argparse.Namespace(
            output_dir=str(Path("/mock/output/dir")),
            num_samples=10,
            force_max=2,
            force_count_max=7,
            batch_seed=12345,
            mode="parallel"
        )
        mock_args.return_value = args
        yield args

def test_main_parallel_mode(mock_training_data_generator, mock_argparse):
    main()
    mock_training_data_generator.generate_parallel.assert_called_once_with(mock_argparse.num_samples)

def test_main_sequential_mode(mock_training_data_generator, mock_argparse):
    mock_argparse.mode = "sequential"
    main()
    mock_training_data_generator.generate_sequential.assert_called_once_with(mock_argparse.num_samples)

def test_main_edge_mode(mock_training_data_generator, mock_argparse):
    mock_argparse.mode = "edge"
    main()
    mock_training_data_generator.generate_edge_cases.assert_called_once_with(mock_argparse.num_samples)

def test_default_arguments(mock_argparse):
    assert mock_argparse.output_dir == str(Path("/mock/output/dir"))
    assert mock_argparse.num_samples == 10
    assert mock_argparse.force_max == 2
    assert mock_argparse.force_count_max == 7
    assert mock_argparse.batch_seed == 12345
    assert mock_argparse.mode == "parallel"