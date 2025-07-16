"""Calculate the elasticity tensor from the displacement field."""

import numpy as np
import ufl.tensors
from fenics import (
    Constant,
    DirichletBC,
    Expression,
    Function,
    FunctionSpace,
    Identity,
    LinearVariationalProblem,
    LinearVariationalSolver,
    Measure,
    Mesh,
    MeshFunction,
    SubDomain,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    div,
    dot,
    dx,
    grad,
    inner,
    near,
)


class ElasticityCalculator:
    """Class to calculate the elasticity tensor from the displacement field."""

    def __init__(
        self,
        mesh: Mesh,
        degree: int,
        lame_function: Function,
        shear_function: Function,
        boundary_tolerance: float,
    ) -> None:
        """Initialize the ElasticityCalculator."""
        self.mesh = mesh
        self.lame_function = lame_function
        self.shear_function = shear_function
        self.boundary_tolerance = boundary_tolerance

        self.displacement_space = VectorFunctionSpace(self.mesh, "P", degree=degree)
        self.displacement_test_function = TestFunction(self.displacement_space)
        self.displacement_trial = TrialFunction(self.displacement_space)
        self.displacement_function = Function(self.displacement_space)

        self.boundary_conditions = self._create_boundary_conditions(
            self.displacement_space, boundary_tolerance
        )

        self.zero_body_force = Constant((0.0, 0.0))

    @staticmethod
    def _create_boundary_conditions(
        displacement_space: VectorFunctionSpace, boundary_tolerance: float
    ) -> DirichletBC:
        """Set up boundary conditions for the simulation.

        This function defines the fixed and roller boundary conditions for the mesh.
        The fixed boundary condition is applied to the bottom left corner,
        while the roller boundary condition is applied to the entire bottom edge.
        """

        def bottom_fixed_boundary(x: np.ndarray, on_boundary: bool) -> bool:
            """Check if the point is on the bottom boundary and in the left corner (x=0, y=0)."""
            return near(x[0], 0, boundary_tolerance) and near(
                x[1],
                0,
                boundary_tolerance,
            )

        boundary_condition_fixed = DirichletBC(
            displacement_space,
            Constant((0.0, 0.0)),
            bottom_fixed_boundary,
            method="pointwise",
        )

        def bottom_roller_boundary(x: np.ndarray, on_boundary: bool) -> bool:
            """Check if the point is on the bottom boundary and apply roller."""
            return near(x[1], 0, boundary_tolerance) and x[0] > 0

        boundary_condition_roller = DirichletBC(
            displacement_space.sub(1),
            Constant(0),
            bottom_roller_boundary,
        )
        return [boundary_condition_fixed, boundary_condition_roller]

    def _setup_subdomains(self) -> None:
        """Set up subdomains for the boundaries of the mesh.

        This function defines the top, right, and left boundaries of the mesh as subdomains
        and marks them with unique identifiers.
        """

        class Top(SubDomain):
            def __init__(self, boundary_tolerance: float) -> None:
                super().__init__()
                self.boundary_tolerance = boundary_tolerance

            def inside(self, x: np.ndarray, on_boundary: bool) -> bool:
                return near(x[1], 1, self.boundary_tolerance) and on_boundary

        class Right(SubDomain):
            def __init__(self, boundary_tolerance: float) -> None:
                super().__init__()
                self.boundary_tolerance = boundary_tolerance

            def inside(self, x: np.ndarray, on_boundary: bool) -> bool:
                return near(x[0], 1, self.boundary_tolerance) and on_boundary

        class Left(SubDomain):
            def __init__(self, boundary_tolerance: float) -> None:
                super().__init__()
                self.boundary_tolerance = boundary_tolerance

            def inside(self, x: np.ndarray, on_boundary: bool) -> bool:
                return near(x[0], 0, self.boundary_tolerance) and on_boundary

        class BoundaryID:
            TOP = 1
            RIGHT = 2
            LEFT = 3

        self.boundaries = MeshFunction("size_t", self.mesh, 1)
        self.boundaries.set_all(0)
        Top(self.boundary_tolerance).mark(self.boundaries, BoundaryID.TOP)
        Right(self.boundary_tolerance).mark(self.boundaries, BoundaryID.RIGHT)
        Left(self.boundary_tolerance).mark(self.boundaries, BoundaryID.LEFT)
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)

    def _setup_force_expression(self) -> None:
        """Set up the force expressions based on the force profile.

        This function builds the force expressions for the top, right, and left boundaries
        based on the provided force profile.
        """
        self.top_force_expr = self._build_force_expression(
            self.force_profile[0],
            axis="x",
        )
        self.right_force_expr = self._build_force_expression(
            self.force_profile[1],
            axis="y",
        )
        self.left_force_expr = self._build_force_expression(
            self.force_profile[2],
            axis="y",
        )

    @staticmethod
    def _build_force_expression(force_row: np.ndarray, axis: str) -> Expression:
        """Build the force expression based on the force profile."""
        expression_pieces = []
        dx = 1.0 / len(force_row)
        for i, value in enumerate(force_row):
            if value != 0:
                start = i * dx
                end = (i + 1) * dx
                condition = (
                    f"{start} <= x[0] && x[0] <= {end}"
                    if axis == "x"
                    else f"{start} <= x[1] && x[1] <= {end}"
                )
                expression_pieces.append(f"({value})*({condition})")
        full_expression = " + ".join(expression_pieces) if expression_pieces else "0.0"
        return Expression(full_expression, degree=1)

    def _initialize_load_form(self) -> None:
        """Initialize the load form for the elasticity problem."""
        self.load_form = (
            dot(self.zero_body_force, self.displacement_test_function) * dx
            + self.displacement_test_function[1] * self.top_force_expr * self.ds(1)
            + self.displacement_test_function[0] * self.right_force_expr * self.ds(2)
            + self.displacement_test_function[0] * self.left_force_expr * self.ds(3)
        )

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

    def _initialize_solver(self) -> None:
        """Initialize the solver for the elasticity problem."""
        problem = LinearVariationalProblem(
            self.stiffness_form,
            self.load_form,
            self.displacement_function,
            self.boundary_conditions,
        )
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "default"
        solver.parameters["preconditioner"] = "hypre_amg"
        solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-10
        solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-10
        solver.parameters["krylov_solver"]["maximum_iterations"] = 1000
        self.elasticity_solver = solver

    def calculate(
        self, displacement: Function, lame_function: Function, shear_function: Function
    ) -> Function:
        """Calculate the elasticity tensor from the displacement field."""
        self.displacement_trial = displacement
        self.lame_function = lame_function
        self.shear_function = shear_function

        self._initialize_stiffness_form()
        self._initialize_load_form()
        self._initialize_solver()

        self.elasticity_solver.solve()

    def reset(self) -> None:
        """Reset the simulation state to the initial conditions."""
        self.shear_function.vector().zero()
        self.lame_function.vector().zero()
        self.current_density = self.initial_density_field[self.mesh_i, self.mesh_j]

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
        self._update_force_expression()
        self._initialize_load_form()
        self._initialize_solver()

    def _update_force_expression(self) -> None:
        """Update the force expressions based on the new force profile."""
        self.top_force_expr = self._build_force_expression(
            self.force_profile[0],
            axis="x",
        )
        self.right_force_expr = self._build_force_expression(
            self.force_profile[1],
            axis="y",
        )
        self.left_force_expr = self._build_force_expression(
            self.force_profile[2],
            axis="y",
        )
