"""Module implements a density simulation for bone remodeling using FEniCS.

It simulates the mechanical behavior of bone tissue under various force profiles,
calculates the strain energy density, and updates the bone density based on the
remodeling rate coefficient and stimulus threshold.

The simulation can be configured with various parameters such as time steps,
density bounds, force profiles, and output settings. It supports saving results
and plotting the final density profile using pyvista.

"""

import logging
from dataclasses import asdict

import numpy as np
from fenics import (  # type: ignore
    Function,
    FunctionSpace,
    LinearVariationalProblem,
    LinearVariationalSolver,
    LogLevel,
    TestFunction,
    UnitSquareMesh,
    VectorFunctionSpace,
    cells,
    set_log_level,
)

from bone_remodelling.forward_model.boundary_condition_builder import (
    BoundaryConditionBuilder,
)
from bone_remodelling.forward_model.calculate_strain_energy_density import (
    StrainEnergyDensityCalculator,
)
from bone_remodelling.forward_model.density_updater import DensityUpdater
from bone_remodelling.forward_model.load_form_builder import (
    LoadFormBuilder,
)
from bone_remodelling.forward_model.parameters import SimulationParameters
from bone_remodelling.forward_model.stiffness_form_builder import (
    StiffnessFormBuilder,
)

logger = logging.getLogger(__name__)


class DensitySimulation:
    """Class for simulating bone density changes under mechanical loads using FEniCS.

    This class implements a forward model for bone remodeling based on the finite element method.
    It simulates the mechanical behavior of bone tissue under various force profiles,
    calculates the strain energy density, and updates the bone density based on the remodeling rate coefficient
    and stimulus threshold.

    """

    force_profile: np.ndarray
    force_mask: np.ndarray
    initial_density_field: np.ndarray
    time_steps: int
    dt: float

    n_rows: int
    n_columns: int

    displacement_element_order: int
    density_element_order: int

    min_density: float
    max_density: float
    poisson_ratio: float
    elastic_modulus_scale: float
    modulus_exponent: float
    stimulus_threshold: float
    convergence_tolerance: float
    convergence_tolerance_decay: float
    convergence_after_steps: int
    convergence_steps_decay: float
    boundary_tolerance: float
    remodeling_rate_coefficient: float
    krylov_solver_tolerance: float
    krylov_solver_iterations: int
    linear_solver: str
    preconditioner: str

    def __init__(self, parameters: SimulationParameters) -> None:
        """Initialize the density simulation with parameters and setup.

        Args:
        ----
            parameters (SimulationParameters): Configuration parameters for the simulation.

        """
        set_log_level(LogLevel.ERROR)

        for name, value in asdict(parameters).items():
            setattr(self, name, value)

        self.n_rows = self.initial_density_field.shape[0]
        self.n_columns = self.initial_density_field.shape[1]

        self._setup_mesh_and_spaces()
        self._setup_density_field(parameters)
        self.boundary_condition_builder = BoundaryConditionBuilder(
            mesh=self.mesh,
            displacement_space=self.displacement_space,
            boundary_tolerance=self.boundary_tolerance,
        )

        self.load_form_builder = LoadFormBuilder(
            mesh=self.mesh,
            force_profile=self.force_profile,
            force_mask=self.force_mask,
            displacement_test_function=self.displacement_test_function,
            boundary_tolerance=self.boundary_tolerance,
        )

        self._initialize_fenics_functions()
        self._update_material_properties()

        self.stiffness_form_builder = StiffnessFormBuilder(
            shear_function=self.shear_function,
            lame_function=self.lame_function,
            displacement_space=self.displacement_space,
            displacement_test_function=self.displacement_test_function,
        )
        self._initialize_solver()

        self.sed_calculator = StrainEnergyDensityCalculator(
            cell_density_space=self.cell_density_space,
            spatial_dimension=self.spatial_dimension,
        )

    def _setup_mesh_and_spaces(self) -> None:
        """Set up the mesh and function spaces for the simulation.

        This function initializes the mesh based on the dimensions of the initial density profile,
        and creates the necessary function spaces for the simulation.
        """
        self.mesh = UnitSquareMesh(self.n_rows, self.n_columns, "left")
        self.displacement_space = VectorFunctionSpace(self.mesh, "P", self.displacement_element_order)
        self.cell_density_space = FunctionSpace(self.mesh, "DG", self.density_element_order)
        self.spatial_dimension = self.displacement_space.ufl_element().value_shape()[0]
        self.displacement_test_function = TestFunction(self.displacement_space)

    def _setup_density_field(self, parameters: SimulationParameters) -> None:
        """Set up the density values for the simulation based on the initial density field.

        This function calculates the centroids of the cells in the mesh and assigns the initial density values
        based on the position of these centroids in relation to the initial density field.
        The centroids are mapped to the corresponding indices in the initial density field,
        ensuring that the density values are correctly assigned to each cell.
        The mapping is done by calculating the indices based on the position of the centroids in the mesh,
        ensuring that they fall within the bounds of the initial density field.
        The current density is initialized to the initial density field values at the corresponding indices,
        and convergence flags are set to false for all cells.
        """
        centroids = np.array([cell.midpoint().array() for cell in cells(self.mesh)])
        xs = centroids[:, 0]
        ys = centroids[:, 1]

        self.mesh_i = np.minimum((ys * self.n_rows).astype(int), self.n_rows - 1)
        self.mesh_j = np.minimum((xs * self.n_columns).astype(int), self.n_columns - 1)

        self.current_density = self.initial_density_field[self.mesh_i, self.mesh_j]

        self.density_updater = DensityUpdater(
            initial_density=self.current_density,
            simulation_parameters=parameters,
        )

    def _initialize_fenics_functions(self) -> None:
        """Initialize reusable objects for the simulation."""
        self.sed_function = Function(self.cell_density_space)
        self.density_function = Function(self.cell_density_space)
        self.elasticity_modulus_function = Function(self.cell_density_space)
        self.shear_function = Function(self.cell_density_space)
        self.lame_function = Function(self.cell_density_space)
        self.displacement = Function(self.displacement_space)

    def _initialize_solver(self) -> None:
        """Initialize the solver for the elasticity problem."""
        load_form = self.load_form_builder.get_load_form()
        stiffness_form = self.stiffness_form_builder.get_stiffness_form()

        problem = LinearVariationalProblem(
            stiffness_form,
            load_form,
            self.displacement,
            self.boundary_condition_builder.get_boundary_conditions(),
        )
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = self.linear_solver
        solver.parameters["preconditioner"] = self.preconditioner
        solver.parameters["krylov_solver"]["relative_tolerance"] = self.krylov_solver_tolerance
        solver.parameters["krylov_solver"]["absolute_tolerance"] = self.krylov_solver_tolerance
        solver.parameters["krylov_solver"]["maximum_iterations"] = self.krylov_solver_iterations
        self.elasticity_solver = solver

    def _update_material_properties(self) -> None:
        """Update the modulus of elasticity, Shear modules and first Lame coefficient (lambda) from the modulus of elasticity."""
        elastic_modulus = self.elastic_modulus_scale * np.power(
            self.current_density,
            self.modulus_exponent,
        )
        shear_modulus = elastic_modulus / (2 * (1 + self.poisson_ratio))
        first_lame_parameter = (elastic_modulus * self.poisson_ratio) / (
            (1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio)
        )

        self.elasticity_modulus_function.vector().set_local(elastic_modulus)
        self.shear_function.vector().set_local(shear_modulus)
        self.lame_function.vector().set_local(first_lame_parameter)

    def _update_density(self) -> None:
        """Compute SED and update density based on it."""
        self.sed_function = self.sed_calculator.calculate(
            displacement=self.displacement,
            lame_function=self.lame_function,
            shear_function=self.shear_function,
        )

        self.current_density = self.density_updater.update(
            self.sed_function.vector().get_local(),
        )
        self.density_function.vector().set_local(self.current_density.copy())

    def step(self) -> None:
        """Run a single step of the simulation.

        This method solves the elasticity problem, updates the density, and checks for convergence.
        It is intended to be called repeatedly to advance the simulation in time steps.
        """
        self.elasticity_solver.solve()
        self._update_density()
        self._update_material_properties()

    def run(self) -> None:
        """Run the full simulation loop."""
        for _ in range(self.time_steps):
            self.step()

            if self.density_updater:
                break

    def reset(self) -> None:
        """Reset the simulation state to the initial conditions."""
        self.displacement.vector().zero()
        self.sed_function.vector().zero()
        self.elasticity_modulus_function.vector().zero()
        self.shear_function.vector().zero()
        self.lame_function.vector().zero()
        self.current_density = self.initial_density_field[self.mesh_i, self.mesh_j]
        self.density_function.vector().set_local(self.current_density.copy())
        self._update_material_properties()
        self.density_updater.reset()

    def update_force_profile(self, new_force_profile: np.ndarray) -> None:
        """Update the force profile for the simulation.

        Args:
        ----
            new_force_profile (np.ndarray): New force profile matrix with shape (3, n).

        """
        if new_force_profile.shape != self.force_profile.shape:
            raise ValueError(
                "New force profile must have the same shape as the original force profile.",
            )
        self.force_profile = new_force_profile
        self.load_form_builder.rebuild(self.force_profile)
        self._initialize_solver()

    def get_density(self) -> np.ndarray:
        """Reconstruct an (n_rows x n_columns) density array by binning the DG0 cell values back onto a structured grid."""
        density_indices = self.mesh_i * self.n_columns + self.mesh_j
        flat_grid = np.bincount(
            density_indices,
            weights=self.current_density,
            minlength=self.n_rows * self.n_columns,
        )
        flat_counts = np.bincount(
            density_indices,
            minlength=self.n_rows * self.n_columns,
        )
        safe_density = np.divide(
            flat_grid,
            flat_counts,
            out=np.zeros_like(flat_grid),
            where=flat_counts != 0,
        )
        return np.flipud(safe_density.reshape(self.n_rows, self.n_columns))

    def get_density_function(self) -> Function:
        """Get the density function used in the simulation.

        Returns
        -------
            Function: The density function used in the simulation.

        """
        return self.density_function
