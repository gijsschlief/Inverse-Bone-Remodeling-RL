import pytest
import torch
import numpy as np

@pytest.fixture
def dummy_force_map():
    return torch.rand(1, 1, 40, 40)

@pytest.fixture
def surrogate_model():
    from surrogate_model.cnn_surrogate import SurrogateModel
    model = SurrogateModel()
    model.eval()
    return model