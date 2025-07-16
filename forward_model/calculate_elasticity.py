"""Calculate the elasticity tensor from the displacement field."""

import numpy as np
import ufl.tensors
from fenics import (
    Constant,
    Expression,
    Function,
    FunctionSpace,
    LinearVariationalProblem,
    LinearVariationalSolver,
    Measure,
    Mesh,
    TestFunction,
    TrialFunction,
    UnitSquareMesh,
    VectorFunctionSpace,
    div,
    dot,
    dx,
    grad,
    inner,
)


class ElasticityCalculator:
    """Class to calculate the elasticity tensor from the displacement field."""

    def __init__(self, mesh: Mesh, displacement_field: Function, boundary_conditions, force_expressions, ds) -> None:
        """Initialize the elasticity calculator with mesh, displacement field, boundary conditions, and force expressions."""
        self.mesh = mesh
        self.displacement_field = displacement_field
        self.boundary_conditions = boundary_conditions
        self.force_expressions = force_expressions
        self.n_rows = mesh.num_cells()[0]
        self.n_columns = mesh.num_cells()[1]

        self._initialize_load_form(force_expressions, ds)

        # Initialize function spaces and other reusable objects
        self._setup_mesh_and_spaces()
        self._initialize_fenics_functions()
        self.force_expression_builder = force_expressions
        self.boundary_condition_builder = boundary_conditions

    def _setup_mesh_and_spaces(self) -> None:
        """Set up the mesh and function spaces for the simulation.

        This function initializes the mesh based on the dimensions of the initial density profile,
        and creates the necessary function spaces for the simulation.
        """
        self.mesh = UnitSquareMesh(self.n_rows, self.n_columns, "left")
        self.displacement_space = VectorFunctionSpace(self.mesh, "P", 3)
        self.cell_density_space = FunctionSpace(self.mesh, "DG", 0)
        self.spatial_dimension = self.displacement_space.ufl_element().value_shape()[0]
        self.displacement_test_function = TestFunction(self.displacement_space)
        self.displacement_trial = TrialFunction(self.displacement_space)
        self.num_cells = self.mesh.num_cells()

    def _initialize_fenics_functions(self) -> None:
        """Initialize reusable objects for the simulation."""
        self.sed_function = Function(self.cell_density_space)
        self.density_function = Function(self.cell_density_space)
        self.elasticity_modulus_function = Function(self.cell_density_space)
        self.shear_function = Function(self.cell_density_space)
        self.lame_function = Function(self.cell_density_space)
        self.displacement = Function(self.displacement_space)


    def _initialize_stiffness_form(self) -> None:
        self.stiffness_form = (
            2
            * self.shear_function
            * inner(
                self._calculate_strain_tensor(self.displacement_trial),
                self._calculate_strain_tensor(self.displacement_test_function),
            )
            * dx
            + self.lame_function
            * dot(div(self.displacement_trial), div(self.displacement_test_function))
            * dx
        )

    @staticmethod
    def _calculate_strain_tensor(displacement: Function) -> ufl.tensors.ListTensor:
        """Calculate the strain tensor from the displacement field."""
        return 0.5 * (grad(displacement) + grad(displacement).T)

    def _initialize_load_form(self, force_expressions: dict[str, Expression], ds: Measure) -> None:
        """Initialize the load form for the elasticity problem."""
        zero_body_force = Constant((0, 0))

        self.load_form = (
            dot(zero_body_force, self.displacement_test_function) * dx
            + self.displacement_test_function[1] * force_expressions["top"] * ds(1)
            + self.displacement_test_function[0] * force_expressions["right"] * ds(2)
            + self.displacement_test_function[0] * force_expressions["left"] * ds(3)
        )

    def _initialize_solver(self) -> None:
        """Initialize the solver for the elasticity problem."""
        problem = LinearVariationalProblem(
            self.stiffness_form,
            self.load_form,
            self.displacement,
            self.boundary_condition_builder.get_boundary_conditions(),
        )
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "default"
        solver.parameters["preconditioner"] = "hypre_amg"
        solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-10
        solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-10
        solver.parameters["krylov_solver"]["maximum_iterations"] = 1000
        self.elasticity_solver = solver
