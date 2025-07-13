"""Module implements a density simulation for bone remodeling using FEniCS.

It simulates the mechanical behavior of bone tissue under various force profiles,
calculates the strain energy density, and updates the bone density based on the
remodeling rate coefficient and stimulus threshold.

The simulation can be configured with various parameters such as time steps,
density bounds, force profiles, and output settings. It supports saving results
and plotting the final density profile using pyvista.

"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ufl  # type: ignore
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

    Attributes
    ----------
        force_profile (np.ndarray): Force profile matrix with shape (3, n) where the rows represent top, left, and right, and n is the location in that row.
        initial_density_field (np.ndarray): Initial bone density matrix.
        time_steps (int): Number of time steps for the simulation.
        dt (float): Time step size.
        parameters (dict[str, Any]): Dictionary containing simulation parameters.
        mesh (UnitSquareMesh): Mesh for the simulation domain.
        displacement_space (VectorFunctionSpace): Function space for displacement.
        cell_density_space (FunctionSpace): Function space for cell density.
        spatial_dimension (int): Spatial dimension of the problem.
        zero_body_force (Constant): Zero body force constant.
        displacement_test_function (TestFunction): Test function for displacement.
        displacement_trial (TrialFunction): Trial function for displacement.
        num_cells (int): Number of cells in the mesh.
        current_density (np.ndarray): Current density values for each cell in the mesh.
        convergence_flags (np.ndarray): Flags indicating whether each cell has converged.
        elasticity_modulus_function (Function): Function representing the modulus of elasticity.
        shear_function (Function): Function representing the shear modulus.
        lame_function (Function): Function representing the first Lame coefficient.
        displacement (Function): Function representing the displacement field.
        boundaries (MeshFunction): Mesh function defining the boundaries of the mesh.
        ds (Measure): Measure for boundary integrals.
        top_force_expr (Expression): Expression for the force applied on the top boundary.
        right_force_expr (Expression): Expression for the force applied on the right boundary.
        left_force_expr (Expression): Expression for the force applied on the left boundary.
        sed_function (Function): Function for the strain energy density.
        density_function (Function): Function for the density field.

    Methods
    -------
        run() -> None:
            Runs the full simulation loop, solving the elasticity problem and updating the density.
        get_density() -> np.ndarray:
            Returns the density profile as a numpy array.
        plot_density() -> None:
            Plots the density profile using pyvista. Only works if pyvista is installed and data is saved.
        _save(to_save_data: Any) -> None:
            Saves specified data to the output directory.
        _check_convergence(time: Optional[float] = None) -> bool:
            Checks if the simulation has converged based on cell convergence flags.
        _update_material_properties() -> None:
            Updates the material properties for the next time step.
        _update_density() -> None:
            Computes the strain energy density and updates the density based on it.
        _calculate_strain_tensor(displacement: Function) -> ufl.tensors.ListTensor:
            Calculates the strain tensor from the displacement field.
        _calculate_stress_tensor(
            displacement: Function,
            strain_tensor: ufl.tensors.ListTensor,
        ) -> ufl.tensors.ListTensor:
            Calculates the stress tensor using the strain tensor, shear modulus, and first Lame coefficient.
        _update_strain_energy_density(
            strain_tensor: ufl.tensors.ListTensor,
            stress_tensor: ufl.tensors.ListTensor,
        ) -> None:
            Calculates the strain energy density (SED) from the strain and stress tensors.
        _update_density_change() -> None:
            Calculates the change in density based on the strain energy density (SED) and updates the density values.
        _build_force_expression(force_row: np.ndarray, axis: str) -> Expression:
            Builds the force expression based on the force profile for a specified axis.
        _setup_mesh_and_spaces() -> None:
            Sets up the mesh and function spaces for the simulation.
        _setup_density_field() -> None:
            Sets up the density values for the simulation based on the initial density field.
        _setup_boundary_conditions() -> None:
            Sets up boundary conditions for the simulation.
        _setup_subdomains() -> None:
            Sets up subdomains for the boundaries of the mesh.
        _setup_force_expression() -> None:
            Sets up the force expressions based on the force profile.
        _initialize_core_parameters(
            force_profile: np.ndarray,
            initial_density_field: np.ndarray,
            time_steps: int,
            dt: float,
        ) -> None:
            Initializes the core simulation parameters by adding them to the class instance.
        _extract_data_from_parameters(parameters: dict[str, Any]) -> None:
            Extracts necessary data from the parameters dictionary to set up the simulation parameters.
        _validate_parameters() -> None:
            Validates the input parameters for the simulation and raises errors for invalid inputs.

    """

    def __init__(
        self,
        force_profile: np.ndarray,
        initial_density_field: np.ndarray,
        time_steps: int = 100,
        dt: float = 1,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the density simulation with parameters and setup.

        Args:
            force_profile (np.ndarray): Force profile matrix with shape (3, n) where the rows represent top, left, and right, and n is the exact location in that row.
            initial_density_field (np.ndarray): Initial bone density matrix.
            time_steps (int, optional): Number of time steps for the simulation.
            dt (float, optional): Time step size.
            parameters (dict[str, Any], optional): Dictionary containing simulation parameters.

        """
        set_log_level(LogLevel.ERROR)
        self._initialize_core_parameters(
            force_profile,
            initial_density_field,
            time_steps,
            dt,
        )
        self._extract_data_from_parameters(parameters or {})

        self._validate_parameters()

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

    def _initialize_core_parameters(
        self,
        force_profile: np.ndarray,
        initial_density_field: np.ndarray,
        time_steps: int,
        dt: float,
    ) -> None:
        """Initialize the core simulation parameters by adding them to the class instance."""
        self.initial_density_field = initial_density_field
        self.force_profile = force_profile
        self.dt = dt
        self.time_steps = time_steps
        self.n_rows = self.initial_density_field.shape[0]
        self.n_columns = self.initial_density_field.shape[1]

    def _extract_data_from_parameters(self, parameters: dict[str, Any]) -> None:
        """Extract necessary data from the parameters dictionary.

        This function is used to set up the simulation parameters.
        """
        self.output_dir = parameters.get("output_dir", "data/fenics")
        self.min_density = parameters.get("min_density", 0.01)
        self.max_density = parameters.get("max_density", 1.74)
        self.boundary_tolerance = parameters.get("boundary_tolerance", 1e-14)
        self.remodeling_rate_coefficient = parameters.get(
            "remodeling_rate_coefficient",
            1,
        )
        self.stimulus_threshold = parameters.get("stimulus_threshold", 0.25)
        self.poisson_ratio = parameters.get("poisson_ratio", 0.3)
        self.elastic_modulus_scale = parameters.get("elastic_modulus_scale", 100)
        self.modulus_exponent = parameters.get("modulus_exponent", 2.0)
        self.output_basename = parameters.get("output_basename", "density_simulation")
        self.file_extension = parameters.get("file_extension", ".pvd")
        self.save_data = parameters.get("save", False)
        self.convergence_tolerance = parameters.get("convergence_tolerance", 1e-6)
        self.convergence_after_steps = parameters.get("convergence_after_steps", 1)

    def _validate_parameters(self) -> None:
        """Validate the input parameters for the simulation."""
        if not isinstance(self.force_profile, np.ndarray):
            raise TypeError("force_profile must be a numpy array.")
        if not isinstance(self.initial_density_field, np.ndarray):
            raise TypeError("initial_density field must be a numpy array.")
        if not isinstance(self.time_steps, int) or self.time_steps <= 0:
            raise ValueError("time_steps must be a positive integer.")
        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            raise ValueError("dt must be a positive number.")
        if not isinstance(self.output_dir, str):
            raise TypeError("Output directory must be a string.")
        if not isinstance(self.min_density, (int, float)):
            raise TypeError("min_density must be a number.")
        if not isinstance(self.max_density, (int, float)):
            raise TypeError("max_density must be a number.")
        if self.min_density < 0 or self.max_density <= self.min_density:
            raise ValueError(
                "r_min must be non-negative and max_density must be greater than min_density.",
            )
        if self.force_profile.shape[0] != 3 or self.force_profile.shape[1] != np.max(
            self.initial_density_field.shape,
        ):
            raise ValueError(
                "force_profile must have 3 rows and columns equal to the maximum of initial_density field dimensions.",
            )
        if np.isnan(self.force_profile).any():
            raise ValueError("force_profile contains NaN values.")
        if np.isnan(self.initial_density_field).any():
            raise ValueError("initial_density field contains NaN values.")
        if (
            not (self.min_density <= self.initial_density_field).all()
            or not (self.initial_density_field <= self.max_density).all()
        ):
            raise ValueError(
                "initial_density field values must be between min_density and max_density.",
            )
        if not (self.dt > 0):
            raise ValueError(
                "dt must be a positive number.",
            )
        if not (self.time_steps <= 1000):
            logging.warning(
                "time_steps is set to a high value, which may lead to long computation times.",
            )
        if not isinstance(self.save_data, bool):
            raise TypeError("save must be a boolean value.")
        if (
            not isinstance(self.convergence_tolerance, (int, float))
            or self.convergence_tolerance <= 0
        ):
            raise ValueError("convergence_tolerance must be a positive number.")
        if self.convergence_tolerance > 0.1:
            logging.warning("convergence tolerance is very large")
        if not isinstance(self.output_basename, str):
            raise TypeError("output_basename must be a string.")
        if not isinstance(self.file_extension, str):
            raise TypeError("file_extension must be a string.")
        if not self.file_extension.startswith("."):
            raise ValueError("file_extension must start with a dot (e.g., '.pvd').")
        if self.n_rows <= 1 or self.n_columns <= 1:
            raise ValueError("n_rows and n_columns must be greater than 1.")
        if not isinstance(self.convergence_after_steps, int) or self.convergence_after_steps <= 0:
            raise ValueError("convergence_after_steps must be a positive integer.")

    def _setup_mesh_and_spaces(self) -> None:
        """Set up the mesh and function spaces for the simulation.

        This function initializes the mesh based on the dimensions of the initial density profile,
        and creates the necessary function spaces for the simulation.
        """
        self.mesh = UnitSquareMesh(self.n_rows, self.n_columns, "left")
        self.displacement_space = VectorFunctionSpace(self.mesh, "P", 1)
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
        self.convergence_flags = np.zeros(self.num_cells, dtype=bool)
        self.convergence_counter = np.zeros(self.num_cells, dtype=int)

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

    def _calculate_strain_tensor(
        self,
        displacement: Function,
    ) -> ufl.tensors.ListTensor:
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

    def _update_strain_energy_density(
        self,
        strain_tensor: ufl.tensors.ListTensor,
        stress_tensor: ufl.tensors.ListTensor,
    ) -> None:
        """Calculate the strain energy density (SED) from the strain and stress tensors."""
        sed_expression = 0.5 * inner(stress_tensor, strain_tensor)
        linear_sed_form = inner(sed_expression, self.test_function_density) * dx
        solve(
            self.strain_energy_density_form == linear_sed_form,
            self.sed_function,
            solver_parameters=self.projection_solver_parameters,
        )

    def _update_density_change(self) -> None:
        """Calculate the change in density based on the strain energy density (SED).

        Cells which have converged will no longer update in the simulation.
        Convergence happens when the lower or upper density is hit or the cell has not made a noticeable change in density.
        """
        strain_energy_density = self.sed_function.vector().get_local()
        active_cells = ~self.convergence_flags
        density = self.current_density
        stimulus = np.zeros_like(density)

        # Compute the stimulus and density change for active cells
        stimulus[active_cells] = (
            strain_energy_density[active_cells] / density[active_cells]
        )
        delta = self.remodeling_rate_coefficient * (stimulus - self.stimulus_threshold)
        density[active_cells] = density[active_cells] + self.dt * delta[active_cells]

        # Clip the new density values to the min and max bounds
        density = np.clip(density, self.min_density, self.max_density)

        # Count cells that have not changed significantly
        cells_converged_small_change = np.abs(delta) < self.convergence_tolerance
        self.convergence_counter[active_cells & cells_converged_small_change] += 1
        self.convergence_counter[active_cells & ~cells_converged_small_change] = 0

        cells_converged_small_change = self.convergence_counter >= self.convergence_after_steps
        self.convergence_flags = (
            self.convergence_flags
            | cells_converged_small_change
        )

        self.density_function.vector().set_local(density.copy())
        self.current_density = density

    def _update_density(self) -> None:
        """Compute SED and update density based on it."""
        strain_tensor = self._calculate_strain_tensor(self.displacement)
        stress_tensor = self._calculate_stress_tensor(self.displacement, strain_tensor)
        self._update_strain_energy_density(strain_tensor, stress_tensor)
        self._update_density_change()

    def save(self, to_save_data: Function | MeshFunction | Expression, output_path: Path | None = None) -> None:
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
            self.full_file_path = output_path / (self.output_basename + self.file_extension)
        else:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            self.full_file_path = str(output_path) + self.file_extension
        try:
            File(str(self.full_file_path)) << to_save_data
        except Exception as e:
            logging.exception(
                f"Failed to save the output to {self.full_file_path}: {e}"
            )
            raise RuntimeError(f"Failed to save the output: {e}") from e

    def _check_convergence(self, time: Optional[float] = None) -> bool:
        """Return True if simulation converged by checking the cells."""
        return bool(np.all(self.convergence_flags))

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
        for time in range(self.time_steps):
            self.step()
            if self._check_convergence(time):
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
        self.convergence_flags.fill(False)
        self.convergence_counter.fill(0)
        self._update_material_properties()

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

    def plot_density(self) -> None:
        """Plot the final density profile using pyvista."""
        if self.save_data is False:
            logging.warning(
                "Plotting is disabled. Set 'save' parameter to True to enable plotting.",
            )
            return
        import pyvista as pv

        try:
            reader = pv.get_reader(self.full_file_path)
            reader.set_active_time_point(0)
        except Exception as e:
            logging.exception(f"Failed to retrieve the file for plotting: {e}")
            return
        try:
            grid = reader.read()[0]
            scalar_field_name = grid.array_names[
                0
            ]  # Automatically get the first scalar field name
            grid.plot(
                scalars=scalar_field_name,
                show_edges=True,
                show_scalar_bar=True,
                clim=[self.min_density, self.max_density],
                cpos="xy",
                show_grid=True,
            )
        except Exception as e:
            logging.exception(f"Failed to plot the result: {e}")
            return


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bone_remodeling.surrogate_model.visualizer import plot_density_matrix
    from matplotlib.animation import FuncAnimation

    SAVE_PATH = "/home/gijs/Desktop/Thesis/data/fenics"
    TIME_STEPS = 100

    gradient_force_profile = np.arange(30).reshape(3, 10) / 50
    gradient_density = np.arange(100).reshape(10, 10) / 100.0 + 0.01
    parameters: dict = {
        "save": True,
        "output_dir": "/home/gijs/Desktop/Thesis/data/fenics",
        "plot": True,
    }
    density_start = np.ones((10, 10)) * 0.8

    simulation = DensitySimulation(
        force_profile=gradient_force_profile,
        initial_density_field=density_start,
        time_steps=TIME_STEPS,
        dt=1.0,
        parameters=parameters,
    )


    density_film: np.ndarray = np.zeros((TIME_STEPS, simulation.n_rows, simulation.n_columns))

    for i in range(TIME_STEPS):
        simulation.step()
        density_film[i] = simulation.get_density()

    fig, ax = plt.subplots()

    def update(frame: int) -> list[plt.Axes]:
        """Update the plot for the current frame."""
        ax.clear()
        plot_density_matrix(
            matrix=density_film[frame, :, :],
            force_profile=gradient_force_profile,
            title=f"Step {frame+1}",
            axis=ax,
        )
        # Return a list of artists for FuncAnimation
        return [ax]

    animation = FuncAnimation(fig, update, frames=TIME_STEPS, interval=100, repeat=True)
    plt.tight_layout()
    plt.show()
