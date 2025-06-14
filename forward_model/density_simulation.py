from pathlib import Path
import logging
from typing import Any

from fenics import *
import numpy as np
import ufl

class DensitySimulation:
    def __init__(self, force_profile: np.ndarray, initial_density: np.ndarray, time_steps: int = 100, dt: int = 1, parameters: dict[str, Any] | None = None) -> None:
        """
        Initialize the density simulation with parameters and setup.
        Parameters:
            force_profile (np.ndarray): Force profile matrix with shape (3, n) where the rows represent top, left and right and n is the exact location in that row.
            initial_density (np.ndarray): Initial bone density matrix.
            time_steps (int): Number of time steps for the simulation.
            dt (Union[int, float]): Time step size.
            parameters (dict): Dictionary containing simulation parameters.
        """
        self._initialize_parameters(force_profile, initial_density, time_steps, dt, parameters)
        self._validate_parameters()
        self._setup_mesh_and_spaces()
        self._initialize_density()
        self._setup_boundary_conditions()
        self._setup_subdomains()
        self._setup_force_expression()
        #self.F = Expression("m*x[0]+c", m=-10, c=10, degree=1)
        self.E0 = self._calculate_E(self.rho_val)
        self.mu, self.lmbda = self._calculate_lame_coefficients(self.E0)
        self.u = Function(self.V)

    def _initialize_parameters(self, force_profile, initial_density, time_steps, dt, parameters) -> None:
        """
        Initialize the simulation parameters and validate them.
        Parameters:
            force_profile (np.ndarray): Force profile matrix with shape (3, n).
            initial_density (np.ndarray): Initial bone density matrix.
            time_steps (int): Number of time steps for the simulation.
            dt (Union[int, float]): Time step size.
            parameters (dict): Dictionary containing simulation parameters.
        """
        self.time_steps = time_steps  # Store time_steps as an instance variable
        # Initialize density and force profiles
        self.density_profile = initial_density
        self.force_profile = force_profile
        self.dt = dt
        self.time_steps = time_steps
        self.T = self.time_steps * self.dt
        self.parameters = parameters

        # Data extraction from the density profile
        self.X = self.density_profile.shape[0]  # Number of rows in initial density
        self.Y = self.density_profile.shape[1]  # Number of columns in initial density
        self.rho0 = self.density_profile.mean()  # Initial average density
        
        # Data extraction from parameters
        default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "fenics"
        self.file_location = parameters.get('file_location', str(default_dir))  # Location of the data files
        self.rho_min = parameters.get('rho_min', 0.01)  # Minimum bone density
        self.rho_max = parameters.get('rho_max', 1.74)  # Maximum bone density
        self.tolerance = parameters.get('tolerance', 1E-14)  # Tolerance for convergence
        self.B = parameters.get('B', 1)  # Coefficient for density change
        self.k = parameters.get('k', 0.25)  # Threshold for density change
        self.nu = parameters.get('nu', 0.3)  # Poisson's ratio
        self.M = parameters.get('M', 100)  # Modulus of elasticity
        self.gamma = parameters.get('gamma', 2.0)  # Exponent for density elasticity
        self.file_name = parameters.get('file_name', 'density_simulation')  # Base name for output files
        self.file_extension = parameters.get('file_extension', '.pvd')  # File extension for output files
        self.save = parameters.get('save', False)  # Flag to save output files
        self.plot = parameters.get('plot', False)  # Flag to plot the results
        self.convergence_eps = parameters.get('convergence_eps', 1E-6)  # Convergence threshold for density change

    def _validate_parameters(self) -> None:
        """
        This function tests if the inputs are valid for the forward model and raises errors for invalid inputs.

        Parameters:
            force_profile (np.ndarray): Force profile matrix with shape (3, n) where the rows represent top, left and right and n is the exact location in that row.
            initial_density (np.ndarray): Initial bone density matrix.
            time_steps (int): Number of time steps for the simulation.
            dt (Union[int, float]): Time step size.
            parameters (dict): Dictionary containing simulation parameters.

        Returns:
            None: If all inputs are valid.
        """
        if not isinstance(self.force_profile, np.ndarray):
            raise TypeError("force_profile must be a numpy array.")
        if not isinstance(self.density_profile, np.ndarray):
            raise TypeError("initial_density must be a numpy array.")
        if not isinstance(self.time_steps, int) or self.time_steps <= 0:
            raise ValueError("time_steps must be a positive integer.")
        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            raise ValueError("dt must be a positive number.")
        if not isinstance(self.file_location, str):
            raise TypeError("file_location must be a string.")
        if not isinstance(self.rho_min, (int, float)):
            raise TypeError("rho_min must be a number.")
        if not isinstance(self.rho_max, (int, float)):
            raise TypeError("rho_max must be a number.")
        if self.rho_min < 0 or self.rho_max <= self.rho_min:
            raise ValueError("rho_min must be non-negative and rho_max must be greater than rho_min.")
        if self.force_profile.shape[0] != 3 or self.force_profile.shape[1] != max(self.density_profile.shape):
            raise ValueError("force_profile must have 3 rows and columns equal to the maximum of initial_density dimensions.")
        if np.isnan(self.force_profile).any():
            raise ValueError("force_profile contains NaN values.")
        if np.isnan(self.density_profile).any():
            raise ValueError("initial_density contains NaN values.")
        if not (self.rho_min <= self.density_profile).all() or not (self.density_profile <= self.rho_max).all():
            raise ValueError("initial_density values must be between rho_min and rho_max.")
        if not (0 < self.dt <= self.time_steps):
            raise ValueError("dt must be a positive number and less than or equal to time_steps.")
        if not (0 <= self.time_steps <= 1000):
            raise ValueError("time_steps must be between 0 and 1000.")
        # If all checks pass, return None indicating no errors
        return None

    def _setup_mesh_and_spaces(self) -> None:
        """
        Setup the mesh and function spaces for the simulation.
        This function initializes the mesh based on the dimensions of the initial density profile,
        and creates the necessary function spaces for the simulation.
        """
        self.mesh = UnitSquareMesh(self.X, self.Y, 'left')
        self.V = VectorFunctionSpace(self.mesh, "P", 1)
        self.V_ele = FunctionSpace(self.mesh, "DG", 0)
        self.d = self.V.ufl_element().value_shape()[0]
        self.f = Constant((0, 0))
        self.v = TestFunction(self.V)
        self.u_trial = TrialFunction(self.V)
        self.cell_count = self.mesh.num_cells()

    def _initialize_density(self) -> None:
        """
        Initialize the density values for the simulation.
        This function sets the initial density values for each cell in the mesh,
        based on the initial density profile provided.
        """
        self.rho_val = [self.rho0 for _ in range(self.mesh.num_cells())]
        self.updated_rho_val = self.rho_val[:]
        self.converged_cell_count = [0 for _ in range(self.mesh.num_cells())]

    def _setup_boundary_conditions(self) -> None:
        """
        Setup boundary conditions for the simulation.
        This function defines the fixed and roller boundary conditions for the mesh.
        The fixed boundary condition is applied to the bottom left corner,
        while the roller boundary condition is applied to the bottom right corner.
        """
        def bottom_fixed_boundary(x, on_boundary) -> bool:
            return near(x[0], 0, self.tolerance) and near(x[1], 0, self.tolerance)

        def bottom_right_boundary(x, on_boundary) -> bool:
            return near(x[1], 0, self.tolerance) and x[0] > 0

        bc_fixed = DirichletBC(self.V, Constant((0., 0.)), bottom_fixed_boundary, method='pointwise')
        bc_roller = DirichletBC(self.V.sub(1), Constant(0), bottom_right_boundary)
        self.bcs = [bc_fixed, bc_roller]

    def _setup_subdomains(self) -> None:
        """
        Setup subdomains for the boundaries of the mesh.
        This function defines the top, right, and left boundaries of the mesh as subdomains
        and marks them with unique identifiers.
        """
        class Top(SubDomain):
            def __init__(self, tolerance: float = 1E-14) -> None:
                super().__init__()
                self.tolerance = tolerance
            def inside(self, x, on_boundary) -> bool:
                return near(x[1], 1, self.tolerance) and on_boundary

        class Right(SubDomain):
            def __init__(self, tolerance: float = 1E-14) -> None:
                super().__init__()
                self.tolerance = tolerance
            def inside(self, x, on_boundary) -> bool:
                return near(x[0], 1, self.tolerance) and on_boundary
            
        class Left(SubDomain):
            def __init__(self, tolerance: float = 1E-14) -> None:
                super().__init__()
                self.tolerance = tolerance
            def inside(self, x, on_boundary) -> bool:
                return near(x[0], 0, self.tolerance) and on_boundary
        

        self.boundaries = MeshFunction('size_t', self.mesh, 1)
        self.boundaries.set_all(0)
        Top().mark(self.boundaries, 1)
        Right().mark(self.boundaries, 2)
        Left().mark(self.boundaries, 3)
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
        expr_pieces = []
        dx = 1.0 / len(force_row)
        for i, val in enumerate(force_row):
            if val != 0:
                start = i * dx
                end = (i + 1) * dx
                cond = f"{start} <= x[0] && x[0] <= {end}" if axis == 'x' else f"{start} <= x[1] && x[1] <= {end}"
                expr_pieces.append(f"({val})*({cond})")
        full_expr = " + ".join(expr_pieces) if expr_pieces else "0.0"
        return Expression(full_expr, degree=1)
    
    def _calculate_E(self, rho_vals: list[float]) -> Function:
        """
        Calculate the modulus of elasticity (E) based on the density values.
        :param rho_vals: List of density values.
        :return: Modulus of elasticity as a Function.
        """
        E_func = Function(self.V_ele)
        E_func.vector().zero()  # Initialize the function vector to zero
        E_array = self.M * np.power(rho_vals, self.gamma)
        E_func.vector().set_local(E_array)
        return E_func

    def _calculate_lame_coefficients(self, E_val: Function) -> tuple[Function, Function]:
        """
        Calculate the Lame coefficients (mu and lambda) from the modulus of elasticity.
        :param E_val: Modulus of elasticity as a Function.
        :return: Shear modulus (mu) and first Lame coefficient (lambda).
        """
        mu = E_val / (2 * (1 + self.nu))
        lmbda = (E_val * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        return mu, lmbda

    def _epsilon(self, u: Function) -> ufl.tensors.ListTensor:
        """
        Calculate the strain tensor from the displacement field.
        :param u: Displacement field.
        :return: Strain tensor.
        """
        return 0.5 * (grad(u) + grad(u).T)

    def _sigma(self, u: Function, mu: Function, lmbda: Function) -> ufl.tensors.ListTensor:
        """
        Calculate the stress tensor using the strain tensor and Lame coefficients.
        :param u: Displacement field.
        :param mu: Shear modulus.
        :param lmbda: First Lame coefficient.
        :return: Stress tensor.
        """
        return lmbda * div(u) * Identity(self.d) + 2 * mu * self._epsilon(u)
    
    def _calculate_sed(self, epsilon_val: ufl.tensors.ListTensor, sigma_val: ufl.tensors.ListTensor) -> tuple[np.ndarray, Function]:
        """
        Calculate the strain energy density (SED) from the strain and stress tensors.
        :param epsilon_val: Strain tensor.
        :param sigma_val: Stress tensor.
        :return: SED as a NumPy array and a Function for visualization.
        """
        SED_val = 0.5 * inner(sigma_val, epsilon_val)
        SED_plot = project(SED_val, self.V_ele)
        return SED_plot.vector().get_local(), SED_plot

    def _calculate_density_change(self, rho_vals: list[float], SED: np.ndarray) -> tuple[Function, list[float], list[int]]:
        """
        Calculate the change in density based on the strain energy density (SED) and update the density values.
        :param rho_vals: Current density values.
        :param SED: Strain energy density values.
        :return: Updated density function, updated density values, and convergence status for each cell.
        """
        rho_plot = Function(self.V_ele)
        rho_array = rho_plot.vector().get_local()
        stimulus = np.zeros_like(rho_array)
        change_in_density: list[float] = []

        for i, SED_val in enumerate(SED):
            if self.converged_cell_count[i] == 0:
                stimulus[i] = SED_val / rho_vals[i]
                change = self.B * (stimulus[i] - self.k)
                new_rho = rho_vals[i] + self.dt * change

                if new_rho <= self.rho_min:
                    new_rho = self.rho_min
                    self.converged_cell_count[i] = 1
                elif new_rho >= self.rho_max:
                    new_rho = self.rho_max
                    self.converged_cell_count[i] = 1
                elif abs(change) < self.convergence_eps:
                    self.converged_cell_count[i] = 1

                rho_array[i] = new_rho
                change_in_density.append(change)
            else:
                rho_array[i] = rho_vals[i]
                stimulus[i] = SED_val / max(rho_vals[i], 1e-8)
                change_in_density.append(0)

        rho_plot.vector().set_local(rho_array)
        return rho_plot, list(rho_array), self.converged_cell_count

    def _solve_elasticity_problem(self) -> None:
        """Solve the elasticity problem for current displacement."""
        a = 2 * self.mu * inner(self._epsilon(self.u_trial), self._epsilon(self.v)) * dx + \
            self.lmbda * dot(div(self.u_trial), div(self.v)) * dx
        L = dot(self.f, self.v) * dx + \
            self.v[1] * self.top_force_expr * self.ds(1) + \
            self.v[0] * self.right_force_expr * self.ds(2) + \
            self.v[1] * self.left_force_expr * self.ds(3)

        solve(a == L, self.u, self.bcs)

    def _update_density(self) -> None:
        """Compute SED and update density based on it."""
        eps_val = self._epsilon(self.u)
        sig_val = self._sigma(self.u, self.mu, self.lmbda)
        SED, _ = self._calculate_sed(eps_val, sig_val)
        rho_function, self.updated_rho_val, self.converged_cell_count = self._calculate_density_change(self.updated_rho_val, SED)
        self.current_rho_function = rho_function  # store for saving/plotting

    def _check_convergence_and_save(self, t: float) -> None:
        """Save intermediate results and check for convergence."""
        if self.save:
            path = Path(self.file_location)
            path.mkdir(parents=True, exist_ok=True)
            File(self.file_location + '/' + self.file_name + self.file_extension) << self.current_rho_function

    def _check_termination(self, t: float) -> bool:
        """Return True if simulation should terminate."""
        return t == self.T or sum(self.converged_cell_count) == self.cell_count

    def _update_material_properties(self) -> None:
        """Update material properties for the next time step."""
        self.E0.assign(self._calculate_E(self.updated_rho_val))
        self.mu, self.lmbda = self._calculate_lame_coefficients(self.E0)

    def run(self) -> None:
        """Run the full simulation loop."""
        t = 0
        while t <= self.T:
            self._solve_elasticity_problem()
            self._update_density()
            self._check_convergence_and_save(t)
            if self._check_termination(t):
                break
    
            self._update_material_properties()
            t += self.dt

    def get_final_density(self) -> np.ndarray:
        """
        Get the final density profile after the simulation.
        :return: Final density profile as a NumPy array.
        """
        half_size = len(self.updated_rho_val) // 2
        return np.array(self.updated_rho_val[:half_size]).reshape((self.X, self.Y))

    def plot_density(self) -> None:
        """
        Plot the final density profile using pyvista.
        """
        if self.save == False:
            logging.warning("Plotting is disabled. Set 'save' parameter to True to enable plotting.")
            return
        import pyvista as pv
        filename = self.file_location + '/' + self.file_name + self.file_extension
        try:
            reader = pv.get_reader(filename)
            reader.set_active_time_point(0)
        except Exception as e:
            logging.error(f"Failed to retrieve the file for plotting: {e}")
            return
        try:
            grid = reader.read()[0]
            scalar_field_name = grid.array_names[0]  # Automatically get the first scalar field name
            grid.plot(scalars=scalar_field_name, show_edges=True, show_scalar_bar=True, clim=[self.rho_min, self.rho_max], cpos='xy', show_grid=True)
        except Exception as e:
            logging.error(f"Failed to plot the result: {e}")
            return
        self.visualize_force_with_pyvista()
        
    def visualize_force_with_pyvista(self):
        """
        Visualize boundary forces using self.force_profile as magnitudes.
        Assumes self.force_profile has shape (3, N), for top, right, and left.
        """
        import pyvista as pv

        mesh = self.mesh
        coords = mesh.coordinates()
        top_nodes = []
        right_nodes = []
        left_nodes = []

        # Classify boundary nodes
        for coord in coords:
            if near(coord[1], 1.0, self.tolerance):  # Top boundary
                top_nodes.append(coord)
            elif near(coord[0], 1.0, self.tolerance):  # Right boundary
                right_nodes.append(coord)
            elif near(coord[0], 0.0, self.tolerance):  # Left boundary
                left_nodes.append(coord)

        # Sort nodes consistently (by x or y) to match force_profile indexing
        top_nodes = sorted(top_nodes, key=lambda x: x[0])    # left to right
        right_nodes = sorted(right_nodes, key=lambda x: -x[1])  # top to bottom
        left_nodes = sorted(left_nodes, key=lambda x: -x[1])   # top to bottom

        # Convert to NumPy arrays
        top_nodes = np.array(top_nodes)
        right_nodes = np.array(right_nodes)
        left_nodes = np.array(left_nodes)

        # Check matching shape
        if (self.force_profile.shape[1] != len(top_nodes) or
            self.force_profile.shape[1] != len(right_nodes) or
            self.force_profile.shape[1] != len(left_nodes)):
            raise ValueError("Mismatch between force_profile columns and boundary node counts.")

        # Construct force vectors
        top_forces = np.column_stack([self.force_profile[0], np.zeros_like(self.force_profile[0])])
        right_forces = np.column_stack([np.zeros_like(self.force_profile[1]), -self.force_profile[1]])
        left_forces = np.column_stack([np.zeros_like(self.force_profile[2]), self.force_profile[2]])

        # Combine all
        all_coords = np.vstack([top_nodes, right_nodes, left_nodes])
        all_forces = np.vstack([top_forces, right_forces, left_forces])

        # Create PyVista objects
        points = pv.PolyData(all_coords)
        points["force"] = all_forces
        arrows = points.glyph(orient="force", scale=False, factor=0.05)


        plotter = pv.Plotter()
        grid = pv.UnstructuredGrid(self.mesh.cells(), self.mesh.cell_types(), self.mesh.coordinates())
        plotter.add_mesh(grid, show_edges=True, opacity=0.3)
        plotter.add_mesh(arrows, color="red", label="Forces")
        plotter.add_legend()
        plotter.show()
