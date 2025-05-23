import pytest

def test_simulator_runs_without_crashing():
    from forward_model import fenics_simulator
    # Assume a method run_simulation(T) exists for testable entry point
    try:
        fenics_simulator.run_simulation(T=10)
    except Exception as e:
        pytest.fail(f"Simulator crashed with error: {e}")
