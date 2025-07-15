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
from pathlib import Path

import numpy as np
import ufl  # type: ignore
from bone_remodeling.forward_model.density_parameters import SimulationParameters
from bone_remodeling.forward_model.density_updater import DensityUpdater
from fenics import (  # type: ignore
    Constant,
    DirichletBC,
    Expression,
    File,
    Function,
    FunctionSpace,
    Identity,
    LinearVariationalProblem,
    LinearVariationalSolver,
    LogLevel,
    Measure,
    MeshFunction,
    SubDomain,
    TestFunction,
    TrialFunction,
    UnitSquareMesh,
    VectorFunctionSpace,
    cells,
    div,
    dot,
    dx,
    grad,
    inner,
    near,
    set_log_level,
    solve,
)


class DensitySimulation:
    """Class for simulating bone density changes under mechanical loads using FEniCS.

    This class implements a forward model for bone remodeling based on the finite element method.
    It simulates the mechanical behavior of bone tissue under various force profiles,
    calculates the strain energy density, and updates the bone density based on the remodeling rate coefficient
    and stimulus threshold.

    """

    force_profile: np.ndarray
    initial_density_field: np.ndarray
    time_steps: int
    dt: float

    n_rows: int
    n_columns: int

    min_density: float
    max_density: float
    poisson_ratio: float
    elastic_modulus_scale: float
    modulus_exponent: float
    stimulus_threshold: float
    convergence_tolerance: float
    convergence_after_steps: int
    boundary_tolerance: float
    remodeling_rate_coefficient: float

    output_dir: str
    full_file_path: str | Path
    output_basename: str
    output_extension: str
    save_data: bool

    def __init__(self, parameters: SimulationParameters) -> None:
        """Initialize the density simulation with parameters and setup.

        Args:
            force_profile (np.ndarray): Force profile matrix with shape (3, n) where the rows represent top, left, and right, and n is the exact location in that row.
            initial_density_field (np.ndarray): Initial bone density matrix.
            time_steps (int, optional): Number of time steps for the simulation.
            dt (float, optional): Time step size.
            parameters (dict[str, Any], optional): Dictionary containing simulation parameters.

        """
        set_log_level(LogLevel.ERROR)

        for name, value in asdict(parameters).items():
            setattr(self, name, value)

        self.n_rows = self.initial_density_field.shape[0]
        self.n_columns = self.initial_density_field.shape[1]

        self._setup_mesh_and_spaces()
        self._setup_density_field()
        self._setup_boundary_conditions()
        self._setup_subdomains()
        self._setup_force_expression()
        self._initialize_fenics_functions()
        self._update_material_properties()
        self._initialize_stiffness_form()
        self._initialize_load_form()
        self._initialize_solver()
        self._initialize_projector()

    def _setup_mesh_and_spaces(self) -> None:
        """Set up the mesh and function spaces for the simulation.

        This function initializes the mesh based on the dimensions of the initial density profile,
        and creates the necessary function spaces for the simulation.
        """
        self.mesh = UnitSquareMesh(self.n_rows, self.n_columns, "left")
        self.displacement_space = VectorFunctionSpace(self.mesh, "P", 3)
        self.cell_density_space = FunctionSpace(self.mesh, "DG", 0)
        self.spatial_dimension = self.displacement_space.ufl_element().value_shape()[0]
        self.zero_body_force = Constant((0, 0))
        self.displacement_test_function = TestFunction(self.displacement_space)
        self.displacement_trial = TrialFunction(self.displacement_space)
        self.num_cells = self.mesh.num_cells()

    def _setup_density_field(self) -> None:
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
            dt=self.dt,
            remodeling_rate_coefficient=self.remodeling_rate_coefficient,
            stimulus_threshold=self.stimulus_threshold,
            min_density=self.min_density,
            max_density=self.max_density,
            convergence_tolerance=self.convergence_tolerance,
            convergence_after_steps=self.convergence_after_steps,
        )


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

    def _initialize_load_form(self) -> None:
        """Initialize the load form for the elasticity problem."""
        self.load_form = (
            dot(self.zero_body_force, self.displacement_test_function) * dx
            + self.displacement_test_function[1] * self.top_force_expr * self.ds(1)
            + self.displacement_test_function[0] * self.right_force_expr * self.ds(2)
            + self.displacement_test_function[0] * self.left_force_expr * self.ds(3)
        )

    def _initialize_solver(self) -> None:
        """Initialize the solver for the elasticity problem."""
        problem = LinearVariationalProblem(
            self.stiffness_form,
            self.load_form,
            self.displacement,
            self.boundary_conditions,
        )
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "default"
        solver.parameters["preconditioner"] = "hypre_amg"
        solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-10
        solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-10
        solver.parameters["krylov_solver"]["maximum_iterations"] = 1000
        self.elasticity_solver = solver

    def _initialize_projector(self) -> None:
        """Initialize the projector for the strain energy density (SED)."""
        trial_function_density = TrialFunction(self.cell_density_space)
        self.test_function_density = TestFunction(self.cell_density_space)
        self.strain_energy_density_form = (
            inner(trial_function_density, self.test_function_density) * dx
        )

        self.projection_solver_parameters = {
            "linear_solver": "cg",
            "preconditioner": "hypre_amg",
            "krylov_solver": {
                "absolute_tolerance": 1e-10,
                "relative_tolerance": 1e-10,
                "maximum_iterations": 1000,
            },
        }

    def _update_material_properties(self) -> None:
        """Update the modulus of elasticity, Shear modules and first Lame coefficient (lambda) from the modulus of elasticity."""
        elastic_modulus = self.elastic_modulus_scale * np.power(
            self.current_density, self.modulus_exponent
        )
        shear_modulus = elastic_modulus / (2 * (1 + self.poisson_ratio))
        first_lame_parameter = (elastic_modulus * self.poisson_ratio) / (
            (1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio)
        )

        self.elasticity_modulus_function.vector().set_local(elastic_modulus)
        self.shear_function.vector().set_local(shear_modulus)
        self.lame_function.vector().set_local(first_lame_parameter)

    @staticmethod
    def _calculate_strain_tensor(displacement: Function) -> ufl.tensors.ListTensor:
        """Calculate the strain tensor from the displacement field."""
        strain_tensor = 0.5 * (grad(displacement) + grad(displacement).T)
        return strain_tensor

    def _calculate_stress_tensor(
        self,
        displacement: Function,
        strain_tensor: ufl.tensors.ListTensor,
    ) -> ufl.tensors.ListTensor:
        """Calculate the stress tensor using the strain tensor, shear_modules and first Lame coefficient (lambda)."""
        stress_tensor = (
            self.lame_function * div(displacement) * Identity(self.spatial_dimension)
            + 2 * self.shear_function * strain_tensor
        )
        return stress_tensor

    def _calculate_strain_energy_density(self) -> None:
        """Calculate the strain energy density (SED) from the strain and stress tensors."""
        strain_tensor = self._calculate_strain_tensor(self.displacement)
        stress_tensor = self._calculate_stress_tensor(self.displacement, strain_tensor)

        sed_expression = 0.5 * inner(stress_tensor, strain_tensor)
        linear_sed_form = inner(sed_expression, self.test_function_density) * dx
        solve(
            self.strain_energy_density_form == linear_sed_form,
            self.sed_function,
            solver_parameters=self.projection_solver_parameters,
        )

    def _update_density(self) -> None:
        """Compute SED and update density based on it."""
        self._calculate_strain_energy_density()

        self.current_density = self.density_updater.update(self.sed_function.vector().get_local())
        self.density_function.vector().set_local(self.current_density.copy())

    def save(
        self,
        to_save_data: Function | MeshFunction | Expression,
        output_path: Path | str | None = None,
    ) -> None:
        """Save data specified to the output_dir.

        Args:
        ----
            to_save_data (Function | MeshFunction | Expression): Data to save, can be a Function, MeshFunction, or Expression.
            output_path (Path | None): Optional path to save the data. If None, uses the default output directory.

        Raises:
        ------
            RuntimeError: If saving fails.

        """
        if output_path is None:
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self.full_file_path = output_path / (
                self.output_basename + self.output_extension
            )
        else:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            self.full_file_path = str(output_path) + self.output_extension
        try:
            File(str(self.full_file_path)) << to_save_data
        except Exception as e:
            logging.exception(
                f"Failed to save the output to {self.full_file_path}: {e}"
            )
            raise RuntimeError(f"Failed to save the output: {e}") from e

    def step(self) -> None:
        """Run a single step of the simulation.

        This method solves the elasticity problem, updates the density, and checks for convergence.
        It is intended to be called repeatedly to advance the simulation in time steps.
        """
        self.elasticity_solver.solve()
        self._update_density()
        self._update_material_properties()
        if self.save_data:
            self.save(self.density_function)

    def run(self) -> None:
        """Run the full simulation loop."""
        for _ in range(self.time_steps):
            self.step()

            if self.density_updater.convergenced():
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

    def get_density(self) -> np.ndarray:
        """Reconstruct an (n_rows x n_columns) density array by binning the DG0 cell values back onto a structured grid."""
        density_indices = self.mesh_i * self.n_columns + self.mesh_j
        flat_grid = np.bincount(
            density_indices,
            weights=self.current_density,
            minlength=self.n_rows * self.n_columns,
        )
        flat_counts = np.bincount(
            density_indices, minlength=self.n_rows * self.n_columns
        )
        safe_density = np.divide(
            flat_grid, flat_counts, out=np.zeros_like(flat_grid), where=flat_counts != 0
        )
        return np.flipud(safe_density.reshape(self.n_rows, self.n_columns))
