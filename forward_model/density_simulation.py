from pathlib import Path
import logging
from typing import Any

from fenics import *
import numpy as np
import ufl

class DensitySimulation:
    def __init__(
        self, 
        force_profile: np.ndarray, 
        initial_density_field: np.ndarray, 
        time_steps: int = 100, 
        dt: int | float = 1, 
        parameters: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize the density simulation with parameters and setup.
        Parameters:
            force_profile (np.ndarray): Force profile matrix with shape (3, n) where the rows represent top, left and right and n is the exact location in that row.
            initial_density_field (np.ndarray): Initial bone density matrix.
            time_steps (int): Number of time steps for the simulation.
            dt (Union[int, float]): Time step size.
            parameters (dict): Dictionary containing simulation parameters.
        """
        self._initialize_core_parameters(force_profile, initial_density_field, time_steps, dt)
        self._initialize_density_parameters()
        self._extract_data_from_parameters(parameters or {})

        self._validate_parameters()

        self._setup_mesh_and_spaces()
        self._setup_density_field()
        self._setup_boundary_conditions()
        self._setup_subdomains()
        self._setup_force_expression()

        self.elastic_modulus_function = self._calculate_E()
        self.shear_modules, self.first_lame_parameter = self._calculate_lame_coefficients(self.elastic_modulus_function)
        self.displacement = Function(self.displacement_space)

    def _initialize_core_parameters(
        self, 
        force_profile: np.ndarray, 
        initial_density_field: np.ndarray, 
        time_steps: int, 
        dt: int | float
    ) -> None:
        """
        Initialize the core simulation parameters by adding them to the class instance.
        """
        self.initial_density_field = initial_density_field
        self.force_profile = force_profile
        self.dt = dt
        self.time_steps = time_steps
        self.total_time = self.time_steps * self.dt

    def _initialize_density_parameters(self) -> None:
        """
        Initialize and extract the density profile from the initial density matrix.
        """
        self.n_rows = self.initial_density_field.shape[0]
        self.n_columns = self.initial_density_field.shape[1]
        self.mean_initial_density = self.initial_density_field.mean()

    def _extract_data_from_parameters(self, parameters: dict[str, Any]) -> None:
        """
        Extract necessary data from the parameters dictionary.
        This function is used to set up the simulation parameters.
        """
        self.output_dir = parameters.get('output_dir', 'data/fenics')
        self.min_density = parameters.get('min_density', 0.01)
        self.max_density = parameters.get('max_density', 1.74)
        self.boundary_tolerance = parameters.get('boundary_tolerance', 1E-14)
        self.remodeling_rate_coefficient = parameters.get('remodeling_rate_coefficient', 1)
        self.stimulus_threshold = parameters.get('stimulus_threshold', 0.25)
        self.poisson_ratio = parameters.get('poisson_ratio', 0.3)
        self.elastic_modulus_scale = parameters.get('elastic_modulus_scale', 100)
        self.modulus_exponent = parameters.get('modulus_exponent', 2.0)
        self.output_basename = parameters.get('output_basename', 'density_simulation')
        self.file_extension = parameters.get('file_extension', '.pvd')
        self.save = parameters.get('save', False)
        self.plot = parameters.get('plot', False)
        self.convergence_tolerance = parameters.get('convergence_tolerance', 1E-6)

    def _validate_parameters(self) -> None:
        """
        This function tests if the inputs are valid for the forward model and raises errors for invalid inputs.
        """
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
            raise ValueError("r_min must be non-negative and max_density must be greater than min_density.")
        if self.force_profile.shape[0] != 3 or self.force_profile.shape[1] != max(self.initial_density_field.shape):
            raise ValueError("force_profile must have 3 rows and columns equal to the maximum of initial_density field dimensions.")
        if np.isnan(self.force_profile).any():
            raise ValueError("force_profile contains NaN values.")
        if np.isnan(self.initial_density_field).any():
            raise ValueError("initial_density field contains NaN values.")
        if not (self.min_density <= self.initial_density_field).all() or not (self.initial_density_field <= self.max_density).all():
            raise ValueError("initial_density field values must be between min_density and max_density.")
        if not (0 < self.dt <= self.time_steps):
            raise ValueError("dt must be a positive number and less than or equal to time_steps.")
        if not (self.time_steps <= 1000):
            logging.warning("time_steps is set to a high value, which may lead to long computation times.")
        if not isinstance(self.save, bool):
            raise TypeError("save must be a boolean value.")
        if not isinstance(self.plot, bool):
            raise TypeError("plot must be a boolean value.")
        if not isinstance(self.convergence_tolerance, (int, float)) or self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be a positive number.")
        if self.convergence_tolerance > 0.1:
            logging.warning("convergence tolerance is very large")
        if not isinstance(self.output_basename, str):
            raise TypeError("output_basename must be a string.")
        if not isinstance(self.file_extension, str):
            raise TypeError("file_extension must be a string.")
        if not self.file_extension.startswith('.'):
            raise ValueError("file_extension must start with a dot (e.g., '.pvd').")
        if self.n_rows <= 1 or self.n_columns <= 1:
            raise ValueError("n_rows and n_columns must be greater than 1.")

    def _setup_mesh_and_spaces(self) -> None:
        """
        Setup the mesh and function spaces for the simulation.
        This function initializes the mesh based on the dimensions of the initial density profile,
        and creates the necessary function spaces for the simulation.
        """
        self.mesh = UnitSquareMesh(self.n_rows, self.n_columns, 'left')
        self.displacement_space = VectorFunctionSpace(self.mesh, "P", 1)
        self.cell_density_space = FunctionSpace(self.mesh, "DG", 0)
        self.spatial_dimension = self.displacement_space.ufl_element().value_shape()[0]
        self.zero_body_force = Constant((0, 0))
        self.displacement_test_function = TestFunction(self.displacement_space)
        self.displacement_trial = TrialFunction(self.displacement_space)
        self.num_cells = self.mesh.num_cells()

    def _setup_density_field(self) -> None:
        """
        Setup the density values for the simulation.
        This function sets the initial density values for each cell in the mesh,
        based on the initial density profile provided.
        """
        # should later be calculated using self.initial_density_field currently estimated based on the mean of the input field.
        self.current_density = np.full(self.num_cells, self.mean_initial_density, dtype=float)
        self.convergence_flags = np.zeros(self.num_cells, dtype=bool)

    def _setup_boundary_conditions(self) -> None:
        """
        Setup boundary conditions for the simulation.
        This function defines the fixed and roller boundary conditions for the mesh.
        The fixed boundary condition is applied to the bottom left corner,
        while the roller boundary condition is applied to the bottom right corner.
        """
        def bottom_fixed_boundary(x, on_boundary) -> bool:
            """
            This function checks if the point is on the bottom boundary and
            in the left corner (x=0, y=0).
            """
            return near(x[0], 0, self.boundary_tolerance) and near(x[1], 0, self.boundary_tolerance)

        def bottom_roller_boundary(x, on_boundary) -> bool:
            """
            This function checks if the point is on the bottom boundary and
            in the right corner (x=1, y=0).
            """
            return near(x[1], 0, self.boundary_tolerance) and x[0] > 0

        boundary_condition_fixed = DirichletBC(self.displacement_space, Constant((0., 0.)), bottom_fixed_boundary, method='pointwise')
        boundary_condition_roller = DirichletBC(self.displacement_space.sub(1), Constant(0), bottom_roller_boundary)
        self.boundary_conditions = [boundary_condition_fixed, boundary_condition_roller]

    def _setup_subdomains(self) -> None:
        """
        Setup subdomains for the boundaries of the mesh.
        This function defines the top, right, and left boundaries of the mesh as subdomains
        and marks them with unique identifiers.
        """
        class Top(SubDomain):
            def __init__(self, boundary_tolerance: float) -> None:
                super().__init__()
                self.boundary_tolerance = boundary_tolerance
            def inside(self, x, on_boundary) -> bool:
                return near(x[1], 1, self.boundary_tolerance) and on_boundary

        class Right(SubDomain):
            def __init__(self, boundary_tolerance: float) -> None:
                super().__init__()
                self.boundary_tolerance = boundary_tolerance
            def inside(self, x, on_boundary) -> bool:
                return near(x[0], 1, self.boundary_tolerance) and on_boundary
            
        class Left(SubDomain):
            def __init__(self, boundary_tolerance: float) -> None:
                super().__init__()
                self.boundary_tolerance = boundary_tolerance
            def inside(self, x, on_boundary) -> bool:
                return near(x[0], 0, self.boundary_tolerance) and on_boundary

        class BoundaryID:
            TOP = 1
            RIGHT = 2
            LEFT = 3
        
        self.boundaries = MeshFunction('size_t', self.mesh, 1)
        self.boundaries.set_all(0)
        Top(self.boundary_tolerance).mark(self.boundaries, BoundaryID.TOP)
        Right(self.boundary_tolerance).mark(self.boundaries, BoundaryID.RIGHT)
        Left(self.boundary_tolerance).mark(self.boundaries, BoundaryID.LEFT)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
    
    def _setup_force_expression(self) -> None:
        """
        Setup the force expressions based on the force profile.
        This function builds the force expressions for the top, right, and left boundaries
        based on the provided force profile.
        """
        self.top_force_expr = self._build_force_expression(self.force_profile[0], axis='x')
        self.right_force_expr = self._build_force_expression(self.force_profile[1], axis='y')
        self.left_force_expr = self._build_force_expression(self.force_profile[2], axis='y')

    @staticmethod
    def _build_force_expression(force_row: np.ndarray, axis: str = 'x') -> Expression:
        """
        Build the force expression based on the force profile.

        Parameters:
            force_row (np.ndarray): Array representing the force values along the specified axis.
            axis (str): Axis along which the force is applied ('x' or 'y').

        Returns:
            Expression: FEniCS Expression object representing the force profile.
        """
        expression_pieces = []
        dx = 1.0 / len(force_row)
        for i, value in enumerate(force_row):
            if value != 0:
                start = i * dx
                end = (i + 1) * dx
                condition = f"{start} <= x[0] && x[0] <= {end}" if axis == 'x' else f"{start} <= x[1] && x[1] <= {end}"
                expression_pieces.append(f"({value})*({condition})")
        full_expression = " + ".join(expression_pieces) if expression_pieces else "0.0"
        return Expression(full_expression, degree=1)
    
    def _calculate_E(self) -> Function:
        """
        Calculate the modulus of elasticity (E) based on the current density values.
        """
        E_function = Function(self.cell_density_space)
        E_function.vector().zero()
        E_array = self.elastic_modulus_scale * np.power(self.current_density, self.modulus_exponent)
        E_function.vector().set_local(E_array)
        return E_function

    def _calculate_lame_coefficients(self, elastic_modulus_function: Function) -> tuple[Function, Function]:
        """
        Calculate the Shear modules and first Lame coefficient (lambda) from the modulus of elasticity.
        """
        shear_modulus = elastic_modulus_function / (2 * (1 + self.poisson_ratio))
        first_lame_parameter = (elastic_modulus_function * self.poisson_ratio) / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))
        return shear_modulus, first_lame_parameter

    def _calculate_strain_tensor(self, displacement: Function) -> ufl.tensors.ListTensor:
        """
        Calculate the strain tensor from the displacement field.
        """
        strain_tensor = 0.5 * (grad(displacement) + grad(displacement).T)
        return strain_tensor

    def _calculate_stress_tensor(
            self, 
            displacement: Function, 
            shear_modules: Function, 
            first_lame_parameter: Function, 
            strain_tensor: ufl.tensors.ListTensor
        ) -> ufl.tensors.ListTensor:
        """
        Calculate the stress tensor using the strain tensor, shear_modules and first Lame coefficient (lambda).
        """
        stress_tensor = first_lame_parameter * div(displacement) * Identity(self.spatial_dimension) + 2 * shear_modules * strain_tensor
        return stress_tensor
    
    def _calculate_sed(self, strain_tensor: ufl.tensors.ListTensor, stress_tensor: ufl.tensors.ListTensor) -> tuple[np.ndarray, Function]:
        """
        Calculate the strain energy density (SED) from the strain and stress tensors.
        """
        SED_value = 0.5 * inner(stress_tensor, strain_tensor)
        SED_plot = project(SED_value, self.cell_density_space)
        return SED_plot.vector().get_local(), SED_plot

    def _calculate_density_change(self, SED: np.ndarray) -> tuple[Function, np.ndarray]:
        """
        Calculate the change in density based on the strain energy density (SED) and update the density values.
        Cells which have converged will no longer update in the simulation.
        Convergence happens when the lower or upper density is hit or the cell has not made a noticeable change in density.
        """
        active_cells = ~self.convergence_flags
        density = self.current_density
        stimulus = np.zeros_like(density)

        # Compute the stimulus and density change for active cells
        stimulus[active_cells] = SED[active_cells] / density[active_cells]
        delta = self.remodeling_rate_coefficient * (stimulus - self.stimulus_threshold)
        density[active_cells] = density[active_cells] + self.dt * delta[active_cells]

        # Clip the new density values to the min and max bounds
        cells_converged_below_min = density <= self.min_density
        cells_converged_above_max = density >= self.max_density
        density = np.clip(density, self.min_density, self.max_density)

        cells_converged_small_change = np.abs(delta) < self.convergence_tolerance
        self.convergence_flags = self.convergence_flags | cells_converged_below_min | cells_converged_above_max | cells_converged_small_change

        # build Fenics Function for density
        density_fenics = Function(self.cell_density_space)
        density_fenics.vector().set_local(density.copy())

        self.current_density = density
        return density_fenics

    def _solve_elasticity_problem(self) -> None:
        """
        Solve the elasticity problem for current displacement.
        """
        # Define the weak form of the elasticity problem
        stiffness_form = 2 * self.shear_modules * inner(self._calculate_strain_tensor(self.displacement_trial), self._calculate_strain_tensor(self.displacement_test_function)) * dx + \
            self.first_lame_parameter * dot(div(self.displacement_trial), div(self.displacement_test_function)) * dx
        
        # Define the linear form for the load
        load_form = dot(self.zero_body_force, self.displacement_test_function) * dx + \
            self.displacement_test_function[1] * self.top_force_expr * self.ds(1) + \
            self.displacement_test_function[0] * self.right_force_expr * self.ds(2) + \
            self.displacement_test_function[1] * self.left_force_expr * self.ds(3)

        solver_parameters={
            'linear_solver': 'default',
            'preconditioner': 'hypre_amg',
            'krylov_solver': {
                'absolute_tolerance': 1E-10,
                'relative_tolerance': 1E-10,
                'maximum_iterations': 1000
            }
        }

        solve(stiffness_form == load_form, self.displacement, self.boundary_conditions, solver_parameters=solver_parameters)

    def _update_density(self) -> Function:
        """
        Compute SED and update density based on it.
        """
        strain_tensor = self._calculate_strain_tensor(self.displacement)
        stress_tensor = self._calculate_stress_tensor(self.displacement, self.shear_modules, self.first_lame_parameter, strain_tensor)
        SED, _ = self._calculate_sed(strain_tensor, stress_tensor)
        density_fenics = self._calculate_density_change(SED)
        return density_fenics

    def _save(self, to_save_data: Any) -> None:
        """
        Save data specified to the output_dir.
        """
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.full_file_path = output_path / (self.output_basename + self.file_extension)
        try:
            File(str(self.full_file_path)) << to_save_data
        except Exception as e:
            logging.error(f"Failed to save the output to {self.full_file_path}: {e}")
            raise RuntimeError(f"Failed to save the output: {e}")

    def _check_convergence(self, time: int = None) -> bool:
        """
        Return True if simulation converged by checking the cells
        """
        if np.all(self.convergence_flags):
            logging.info("Simulation converged early at time step %s", time)
            return True
        return False

    def _update_material_properties(self) -> None:
        """
        Update material properties for the next time step.
        """
        self.elastic_modulus_function.assign(self._calculate_E())
        self.shear_modules, self.first_lame_parameter = self._calculate_lame_coefficients(self.elastic_modulus_function)

    def run(self) -> None:
        """
        Run the full simulation loop.
        """
        time = 0
        while time <= self.total_time:
            self._solve_elasticity_problem()
            density_fenics = self._update_density()
            if self._check_convergence(time):
                break
    
            self._update_material_properties()
            time += self.dt
        
        if self.save:
            self._save(density_fenics)

    def get_final_density(self) -> np.ndarray:
        """
        Get the final density profile after the simulation. 
        Divided by two as Fenics works with triangular elements and the needed datatype is rectangular.
        """
        half_size = len(self.current_density) // 2
        return np.array(self.current_density[:half_size]).reshape((self.n_rows, self.n_columns))

    def plot_density(self) -> None:
        """
        Plot the final density profile using pyvista.
        """
        if self.save is False:
            logging.warning("Plotting is disabled. Set 'save' parameter to True to enable plotting.")
            return
        import pyvista as pv
        try:
            reader = pv.get_reader(self.full_file_path)
            reader.set_active_time_point(0)
        except Exception as e:
            logging.error(f"Failed to retrieve the file for plotting: {e}")
            return
        try:
            grid = reader.read()[0]
            scalar_field_name = grid.array_names[0]  # Automatically get the first scalar field name
            grid.plot(scalars=scalar_field_name, show_edges=True, show_scalar_bar=True, clim=[self.min_density, self.max_density], cpos='xy', show_grid=True)
        except Exception as e:
            logging.error(f"Failed to plot the result: {e}")
            return
        
