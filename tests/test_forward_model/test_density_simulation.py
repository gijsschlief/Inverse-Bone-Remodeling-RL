import pytest
import numpy as np
from fenics import *
from ufl.core.expr import Expr

from forward_model.density_simulation import DensitySimulation

@pytest.fixture
def simulation_parameters():
    force_profile = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    initial_density = np.ones((4, 4))
    time_steps = 10
    dt = 0.1
    parameters = {
        'rho_min': 0.1,
        'rho_max': 1.5,
        'tolerance': 1E-14,
        'B': 0.1,
        'k': 0.01,
        'nu': 0.3,
        'M': 1.0,
        'gamma': 2.0,
        'save': False
    }
    return force_profile, initial_density, time_steps, dt, parameters


@pytest.fixture
def density_simulation(simulation_parameters):
    force_profile, initial_density, time_steps, dt, parameters = simulation_parameters
    return DensitySimulation(force_profile, initial_density, time_steps, dt, parameters)


def test_initialize_parameters(density_simulation, simulation_parameters):
    _, initial_density, time_steps, dt, parameters = simulation_parameters
    assert density_simulation.time_steps == time_steps
    assert density_simulation.density_profile.shape == initial_density.shape
    assert density_simulation.dt == dt
    assert density_simulation.parameters == parameters


def test_setup_mesh_and_spaces(density_simulation):
    density_simulation.setup_mesh_and_spaces()
    assert isinstance(density_simulation.mesh, UnitSquareMesh)
    assert isinstance(density_simulation.V, FunctionSpace)
    assert isinstance(density_simulation.V_ele, FunctionSpace)
    assert density_simulation.mesh.num_cells() > 0


def test_initialize_density(density_simulation):
    density_simulation.initialize_density()
    assert len(density_simulation.rho_val) == density_simulation.mesh.num_cells()
    assert all(rho == density_simulation.rho0 for rho in density_simulation.rho_val)


def test_setup_boundary_conditions(density_simulation):
    density_simulation.setup_boundary_conditions()
    assert len(density_simulation.bcs) == 2


def test_setup_subdomains(density_simulation):
    density_simulation.setup_subdomains()
    assert density_simulation.boundaries is not None
    assert isinstance(density_simulation.ds, Measure)


def test_build_force_expression():
    force_row = [0, 1, 0, 2]
    expr = DensitySimulation.build_force_expression(force_row, axis='x')
    assert isinstance(expr, Expression)


def test_calculate_E(density_simulation):
    rho_vals = [1.0, 0.5, 0.2]
    density_simulation.rho_val = rho_vals
    density_simulation.mesh = UnitSquareMesh(1, 2)  # 2 cells
    density_simulation.V_ele = FunctionSpace(density_simulation.mesh, "DG", 0)

    E_func = density_simulation.calculate_E(rho_vals)
    assert isinstance(E_func, Function)
    assert len(E_func.vector().get_local()) == density_simulation.mesh.num_cells()

def test_calculate_lame_coefficients(density_simulation):
    E_val = Function(density_simulation.V_ele)
    E_val.vector().set_local(np.array([1.0, 0.5, 0.2]))
    mu, lmbda = density_simulation._calculate_lame_coefficients(E_val)
    assert isinstance(mu, Expr)
    assert isinstance(lmbda, Expr)


def test_run_simulation(density_simulation):
    density_simulation.run()
    assert len(density_simulation.updated_rho_val) == density_simulation.mesh.num_cells()


def test_get_final_density(density_simulation):
    density_simulation.run()
    final_density = density_simulation.get_final_density()
    assert final_density.shape == (density_simulation.X, density_simulation.Y)

if __name__ == "__main__":
    pytest.main([__file__])