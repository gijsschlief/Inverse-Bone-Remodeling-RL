import numpy as np
import pytest

from forward_model.data_generation import serialize_data
    
def test_serialize_data():
    """Test the serialize_data function."""
    serial_number = 1
    force_profile = np.array([[0.5, -0.2], [0.1, 0.0]])
    result = np.array([[1.0, 1.0], [1.0, 1.0]])
    error = "Test error"

    serialized = serialize_data(serial_number, force_profile, result, error)

    assert serialized["serial_number"] == serial_number
    assert serialized["force_profile"] == force_profile.tolist()
    assert serialized["final_output_density"] == result.tolist()
    assert serialized["error"] == error

def test_serialize_data_no_error():
    """Test serialize_data when no error is provided."""
    serial_number = 1
    force_profile = np.array([[0.5, -0.2], [0.1, 0.0]])
    result = np.array([[1.0, 1.0], [1.0, 1.0]])

    serialized = serialize_data(serial_number, force_profile, result)

    assert serialized["serial_number"] == serial_number
    assert serialized["force_profile"] == force_profile.tolist()
    assert serialized["final_output_density"] == result.tolist()
    assert "error" not in serialized

if __name__ == "__main__":
    pytest.main([__file__])