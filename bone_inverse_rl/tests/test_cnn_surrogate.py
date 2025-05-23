import torch
from surrogate_model.cnn_surrogate import SurrogateModel

def test_surrogate_output_shape():
    model = SurrogateModel()
    model.eval()

    # Dummy input: batch of 1 force map (e.g., 1 channel, 40x40)
    x = torch.randn(1, 1, 40, 40)
    with torch.no_grad():
        y = model(x)
    
    assert y.shape == (1, 1, 40, 40), "Output shape should match input"
    assert torch.isfinite(y).all(), "Output should contain only finite values"
