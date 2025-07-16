"""Builds load forms for the forward model of bone remodeling."""

import numpy as np
import ufl  # type: ignore
from fenics import (  # type: ignore
    Constant,
    Expression,
    Measure,
    Mesh,
    MeshFunction,
    SubDomain,
    TestFunction,
    dot,
    dx,
    near,
)


class LoadFormBuilder:
    """The LoadFormBuilder class constructs load forms for the boundaries of the mesh.

    It uses the force profile provided by the ForceProfileGenerator to create expressions
    that can be used in the finite element method for simulating bone remodeling.

    Methods
    -------
        force_expression: Returns the force expressions for the boundaries.
        rebuild: Rebuilds the force expressions with a new force profile.

    """

    def __init__(
        self, mesh: Mesh, force_profile: np.ndarray, displacement_test_function: TestFunction, boundary_tolerance: float = 1e-6
    ) -> None:
        """Initialize the ForceExpressionBuilder with a mesh and force profile.

        Args:
        ----
            mesh (Mesh): The mesh on which the force expressions will be defined.
            force_profile (np.ndarray): The force profile to be used for building the expressions.
            displacement_test_function (TestFunction): The test function for the displacement.
            boundary_tolerance (float): Tolerance for defining the boundaries.

        """
        self.mesh = mesh
        self.force_profile = force_profile
        self.boundary_tolerance = boundary_tolerance
        self.displacement_test_function = displacement_test_function
        self._setup_subdomains()
        self._setup_force_expression()
        self._setup_load_form()

    def get_load_form(self) -> ufl.form.Form:
        """Return the load form for the boundaries."""
        return self.load_form

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
        top_force_expression = self._build_one(
            self.force_profile[0],
            axis="x",
        )
        right_force_expression = self._build_one(
            self.force_profile[1],
            axis="y",
        )
        left_force_expression = self._build_one(
            self.force_profile[2],
            axis="y",
        )

        self.force_expressions = {
            "top": top_force_expression,
            "right": right_force_expression,
            "left": left_force_expression,
        }

    @staticmethod
    def _build_one(force_row: np.ndarray, axis: str) -> Expression:
        """Build one force expression based on the force profile."""
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

    def _setup_load_form(self) -> None:
        """Initialize the load form for the elasticity problem."""
        zero_body_force = Constant((0, 0))

        self.load_form = (
            dot(zero_body_force, self.displacement_test_function) * dx
            + self.displacement_test_function[1] * self.force_expressions["top"] * self.ds(1)
            + self.displacement_test_function[0] * self.force_expressions["right"] * self.ds(2)
            + self.displacement_test_function[0] * self.force_expressions["left"] * self.ds(3)
        )

    def rebuild(self, force_profile: np.ndarray) -> None:
        """Rebuild the load form with a new force profile.

        Args:
        ----
            force_profile (np.ndarray): The new force profile to use for rebuilding the expressions.

        """
        self.force_profile = force_profile
        self._setup_force_expression()
        self._setup_load_form()
