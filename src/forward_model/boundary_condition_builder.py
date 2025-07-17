"""Builds boundary conditions for the bone remodeling simulation."""

import numpy as np
from fenics import (  # type: ignore
    Constant,
    DirichletBC,
    Mesh,
    VectorFunctionSpace,
    near,
)


class BoundaryConditionBuilder:
    """Builds boundary conditions for the bone remodeling simulation."""

    def __init__(
        self,
        mesh: Mesh,
        displacement_space: VectorFunctionSpace,
        boundary_tolerance: float = 1e-6,
    ) -> None:
        """Initialize the BoundaryConditionBuilder with a mesh and boundary tolerance."""
        self.mesh = mesh
        self.displacement_space = displacement_space
        self.boundary_tolerance = boundary_tolerance
        self.boundary_conditions: list[DirichletBC] = []
        self._setup_boundary_conditions()

    def _setup_boundary_conditions(self) -> None:
        """Set up boundary conditions for the simulation.

        This function defines the fixed and roller boundary conditions for the mesh.
        The fixed boundary condition is applied to the bottom left corner,
        while the roller boundary condition is applied to the entire bottom edge.
        """

        def bottom_fixed_boundary(x: np.ndarray, on_boundary: bool) -> bool:
            """Check if the point is on the bottom boundary and in the left corner (x=0, y=0)."""
            return near(x[0], 0, self.boundary_tolerance) and near(
                x[1],
                0,
                self.boundary_tolerance,
            )

        def bottom_roller_boundary(x: np.ndarray, on_boundary: bool) -> bool:
            """Check if the point is on the bottom boundary and apply roller."""
            return near(x[1], 0, self.boundary_tolerance) and x[0] > 0

        boundary_condition_fixed = DirichletBC(
            self.displacement_space,
            Constant((0.0, 0.0)),
            bottom_fixed_boundary,
            method="pointwise",
        )
        boundary_condition_roller = DirichletBC(
            self.displacement_space.sub(1),
            Constant(0),
            bottom_roller_boundary,
        )
        self.boundary_conditions = [boundary_condition_fixed, boundary_condition_roller]

    def get_boundary_conditions(self) -> list[DirichletBC]:
        """Return the list of boundary conditions."""
        return self.boundary_conditions
